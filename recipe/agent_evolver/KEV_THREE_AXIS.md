# kev 接进三轴（lens × lever × intent）：软/硬两种替代

> 目标模型 Qwen3.5-4B · meta 模型 deepseek-v4-pro · 只进化 harness（权重永不训练）。
> 本文记录把 kev（System-1 离散决策模型）从「benchmark 专属闭集词表」升级为「框架三轴词表 + 软/硬两种替代」的实现。
> 配套阅读：`META_AGENT_DECISIONS.md`（LLM 14 个决断 + JEV 可替代性分析）、`EVOLUTION.md`（方法/结果）、`harnessx/decision/`（JudgeQ/ChooseQ 通用接口）。

---

## 0. 结论先行

上一版 kev 接入（Mode A）用 `find/acquire/malformed`（ALF）/ `collapse/hole/...`（gridgames）这套**benchmark 专属闭集词表**给失败轨迹打分。问题：不 general——只对 ALF 有用，是对框架自身「开集三轴诊断」的退化。

本版做两件事：

1. **通用化词表**：kev 的打分词表换成 `analyze` skill 自己定义的**三轴**（lens 看什么 × lever 改什么 × intent 意图），本就 benchmark-agnostic。
2. **软 + 硬两种替代，参数控制**：`--decision-mode {prior, enforce}`。

- **`prior`（软先验）**：kev 打分 → 注入 brief，deepseek 可覆盖（= 旧 Mode A 语义，词表换成三轴）。
- **`enforce`（硬替代）**：kev 的 lever 决策是**绑定的**（事后 gate 检查）；candidate 的 intent 做**retroactive-check 打分**，判「no」的 candidate 拒绝该轮。

| 决策（对应 `META_AGENT_DECISIONS.md`） | 三轴 | 模式 | 载体 |
|---|---|---|---|
| #2 诊断+归因（lens） | lens：Failure / Capability-gap | prior | `decision_priors.md` 注入 brief |
| #5 选杠杆（lever） | lever：Config/Control/Action/Instruction | enforce | `_decision_lever` gate |
| #4 反事实校验（intent） | intent：corrective / preservative-lock / transfer | enforce | `candidate_retrocheck` gate |

---

## 1. 三轴词表（benchmark-agnostic）

### 轨迹级（evolve 前，打分失败轨迹）—— lens × lever

定义在 `recipe/agent_evolver/decision_prior.py`：

- `LENS_FAILURE_Q`（JudgeQ）：是否被 blocker 卡住没跑完（budget/loop/error/commit 类机械/承诺失败）。
- `LENS_CAPABILITY_GAP_Q`（JudgeQ）：是否尝试了但能力/工具/知识做不到。
- `LEVER_Q`（ChooseQ）：修复该归到哪个杠杆，options = `configuration / control / action / instruction`（对齐 `journal._VALID_LEVERS`）。

`AXIS_QUESTIONS = [LENS_FAILURE_Q, LENS_CAPABILITY_GAP_Q, LEVER_Q]`，`AXIS_NAMES = ["lens_failure", "lens_capability_gap", "lever"]`。

> **lens 只做两问（Failure / Capability-gap），不做 Success lens**——因为轨迹级 prior 只读失败轨迹（parser 跳过 `eval_reason=="won"`），Success lens 在轨迹级无对象；它只在 candidate 级的 preservative-lock/transfer intent 里出场（见 §1.2）。

### candidate 级（evolve 后，打分 intent）—— retroactive-check 变体

定义在 `recipe/agent_evolver/candidate_retrocheck.py::retro_question(intent)`，把 intent 映射到 `analyze` 的反事实校验变体：

| intent | retroactive 变体 | JudgeQ 题 |
|---|---|---|
| corrective / preservative-transfer | Variant A / C | 「给定该 candidate 改动，它针对的失败轨迹会不会翻正？」 |
| preservative-lock | Variant B | 「移除该习惯，它针对的成功轨迹会不会翻挂？」 |

---

## 2. 两种模式（`--decision-mode`）

- `--decision-backend {llm, kev, ...}`（已有）：`llm`=关（原路径，meta-LLM 自己做全部诊断）；非 `llm`=注入该决策模型。
- `--decision-base-url`（已有）：sidecar 地址，默认 `http://127.0.0.1:8090`。
- `--decision-mode {prior, enforce}`（新增，默认 `prior`）。

| 模式 | lens | lever | intent | 失败时的语义 |
|---|---|---|---|---|
| `prior`（软） | 注入先验 | 注入先验（可覆盖） | 不启用 | 后端挂了 → warn + 回退 llm |
| `enforce`（硬） | 注入先验 | **绑定 + gate** | **retrocheck + gate** | gate 不过 → 整轮 revert（reuse 当前 config） |

---

## 3. 实现（三阶段）

### Phase 1 — 通用词表 + 类型感知渲染 + 软先验

- `harnessx/meta_harness/decision_prior.py`
  - `analyze_trajectories(questions: Sequence[Question])` 放宽类型（原写死 `Sequence[JudgeQ]`）。
  - `render_priors_markdown` 按 `isinstance(q, ChooseQ)` 分流：JudgeQ 列渲染 `p(true)`，ChooseQ 列渲染 `argmax (conf)`。
  - 新增 `aggregate_choices(rows, idx) -> Decision`：某 ChooseQ 列按 name 求 mean 概率 → argmax + confidence（enforce 读绑定杠杆用）。
- `recipe/agent_evolver/decision_prior.py`
  - 删 `ALF_QUESTIONS/GRIDGAMES_QUESTIONS`，新增三轴 `AXIS_QUESTIONS/AXIS_NAMES`。
  - `_BENCHMARKS` → `_PARSERS`（benchmark → `(parse, render)` 对）：ALF/gridgames 复用 `**Extracted action**` parser；WebShop 显式 `NotImplementedError`（现状即如此）。
  - parser 扩展返回 `exit_reason/steps`，`_render_state` 一并渲染（否则 lever 判断缺信号）。
  - `build_decision_priors` 返回 `DecisionPriors` dataclass：`{path, lever_argmax, lever_confidence, lens_failure_p, lens_capability_gap_p}`。

### Phase 2 — 硬杠杆替代（enforce 的 lever 部分）

- `recipe/agent_evolver/run.py`：`--decision-mode enforce` 时从 `DecisionPriors` 取 `lever_argmax/confidence`，透传 `decision_lever` 给 `evolve()`。
- `harnessx/meta_harness/agent.py`
  - `evolve(..., decision_lever=None, decision_lever_confidence=None)`；`_prepare_brief_and_context` / `_render_task_brief` 透传。
  - `_render_task_brief`：`decision_lever` 非空时加绑定行 ——「本轮 lever 已由 System-1 决定为 `<lever>`（conf=…），**在此杠杆内创作，勿重选**；candidates.md 的 `lever:` tag 与 journal 的 `levers` 必须等于 `<lever>`。」
- `harnessx/meta_harness/validate_workflow.py`
  - `EvolveValidator.run(..., decision_lever=None)`；policy 段（diff 非空时）加 `_decision_lever`：解析 `candidates.md` 的 `lever:` tag + `latest_entry().levers`，任一 ≠ 绑定值 → 写 `DECISION_LEVER_FAIL.md` + `RuntimeError`（run.py 侧 revert）。

### Phase 3 — intent 轴（candidate 级 retroactive-check）

- 新增 `recipe/agent_evolver/candidate_retrocheck.py`
  - `parse_candidates(path)`：按 `## Candidate C-NNN` 切块 + 解析 `[lens | lever | intent]` tag（同步、可单测）。
  - `retro_question(intent)`：intent → retroactive 变体 JudgeQ（§1.2）。
  - `build_candidate_retrocheck(...)`：每个 candidate 打一个 JudgeQ（`pass_prob = p(true)`），写出 `candidate_retrocheck.md`，返回 `CandidateRetrocheck`（`rejected_ids` = `pass_prob < 0.5`）。
- `recipe/agent_evolver/run.py`
  - evolve 成功后（`status == "ok"` 且 enforce + 非 llm）调用 retrocheck。
  - `_cited_rejected(retro, memo_path)`：**只 gate 被 journal `cited_candidates` 引用的 candidate**（读不到 journal 则 gate 全部，保守）。
  - 有拒绝 → `next_evolve_status = "rejected_retrocheck"`、`current_config = pre_evolve_config`（revert）；journal 回填映射成 `reverted`。
  - 后端挂了 / 无 candidate → 非致命 skip，回退 llm。

### 偏离计划的一处

计划 Phase 3 第 8 步写「`validate_workflow.py` 加 `_decision_retrocheck`」。实际**没加**，retrocheck gate 整个放 run.py 侧。原因：retrocheck 是 async + 要带 kev 后端，而 `EvolveValidator` 在 `evolve()` 内部跑、得把 model/base_url 一路透传进去；run.py 侧 revert 更简单且效果等价。同时 retrocheck 只读 candidate body 做接地（body 已含 meta-LLM 蒸馏的轨迹证据），不去解析 "Tasks affected" 的散文式 task id（太脆）。

---

## 4. 改动文件

| 文件 | 改动 |
|---|---|
| `recipe/agent_evolver/decision_prior.py` | 三轴词表 + `_PARSERS` + `DecisionPriors` |
| `harnessx/meta_harness/decision_prior.py` | `Sequence[Question]` + 类型感知渲染 + `aggregate_choices` |
| `recipe/agent_evolver/run.py` | `--decision-mode` + `_cited_rejected` + evolve 分支透传 + retrocheck 段 + enforce 硬 fail guard（后端挂→`raise` 而非静默回退）+ pre-flight kev 探活 |
| `harnessx/meta_harness/agent.py` | `evolve(decision_lever, decision_lever_confidence)` + brief 绑定行 |
| `harnessx/meta_harness/validate_workflow.py` | `_decision_lever` gate |
| `recipe/agent_evolver/candidate_retrocheck.py` | **新增** intent retrocheck |

---

## 5. 验证状态

**已过（单元/冒烟，无 infra）**：
- 全部 touched 文件 `py_compile` 通过。
- `parse_candidates` / `retro_question`（四种 intent 映射）/ 空 candidate 非致命 / `rejected_ids`。
- `_cited_rejected`：no-memo、cite-good、cite-bad、cite-both 四分支（真实 `<!-- journal:frontmatter -->` 格式）。
- Phase 1 三轴 renderer（JudgeQ 列 `0.62`、ChooseQ 列 `instruction (0.47)`）+ `aggregate_choices`。
- `--decision-mode` 参数解析（默认 `prior`，`enforce` 合法）。

**未真跑（风险）**：
- intent retrocheck 的「离线判断 → rollout 成功率」是否可靠**未验证**——历史上离线 Δlogp 不预测 rollout 成功率（记忆 `harnessx-nextbatch-migration-failed`）。
- lever 4-way 粒度粗：ALF/WS/gridgames 是裸 ReAct 文本 loop，杠杆几乎总是 instruction/configuration，硬 gate 可能近乎恒真、增量有限。lens（failure vs capability-gap）和 intent（would-it-have-passed）信号更足。

---

## 6. 下一步

1. **小样本先验** intent retrocheck 是否可靠，再决定 gate 阈值（当前保守 `0.5`）是保持硬 reject 还是降级 advisory。
2. **三 arm temp=0 对照**（脚本已写好 `start_three_axis_infra.sh <bench>` + `three_axis_q0.sh <bench>`，`<bench>` = `alfworld|webshop`，默认 alfworld）：`--decision-mode prior` vs `enforce` vs `llm`，64×6 轮 + held-out 探针，看 enforce 是否 ≥ prior ≥ llm 且不过拟合。启动：`setsid nohup bash three_axis_q0.sh alfworld >> /tmp/three_axis_q0.log 2>&1 &`。enforce arm 已在 run.py 里做 pre-flight + 硬 fail，kev 挂会 loud abort 而非静默退化成 prior/llm。
3. 若 lever gate 测下来是噪音 → 降级 advisory（只做 lens + intent 两轴硬 gate）。

---

## 7. 接口参考

- `harnessx/decision/types.py`：`JudgeQ`(noul) / `ChooseQ`(argmax over `options: tuple[(name,desc),...]`) / `Question = Union[JudgeQ, ChooseQ]` / `Decision(probabilities, argmax, confidence)`，`for_judge` / `for_choose` 工厂。
- `harnessx/decision/kev.py::KevBackend`：`POST /v1/systemone`，原生支持 noul + choice；构造 `base_url/model/timeout/limits`。
- `harnessx/decision/registry.py::get_decision_model(name, **kwargs)`：后端注册表，默认 `{"kev": KevBackend}`。
- 杠杆合法性复用 `journal._VALID_LEVERS`；candidates.md 解析复用 evidence gate 的 `^##\s+Candidate\s+(C-\d+)\b` header 正则。
