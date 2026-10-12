# Harness-JEV：只进化 harness 的 agent 进化器（+ kev System-1 先验注入）

> **核心主张：换 harness，不换模型。** agent 权重**永不训练**——被进化的只有 `system_prompt` / `guidance` / 历史窗口 / loop 参数；一个冻结的 System-1 判别读头（kev）给强 meta-agent 注入「失败机制」先验。
>
> agent 模型 = **Qwen3.5-4B**（冻结，vLLM 服务）· meta-agent = **deepseek-v4-pro**（读轨迹、写 guidance）· kev = **Qwen3.5-4B-Base + LoRA + pointer head**（冻结读头，一次前向打分）。
>
> 本文档讲清**方法 + 如何复现**。方法/结果的全量详述见 [EVOLUTION.md](EVOLUTION.md)（纯 harness 进化）、[KEV_THREE_AXIS.md](KEV_THREE_AXIS.md)（三轴 kev 先验）、[META_AGENT_DECISIONS.md](META_AGENT_DECISIONS.md)（meta 决断清单）。仓库级总表见根 [RESULTS.md](../../RESULTS.md) 与 [REPORT_visibility.md](../../REPORT_visibility.md)。

---

## 1. 这是什么

一个**文本协议进化器**，用同一套 `run.py` 做两件事：

1. **纯 harness 进化**：meta-agent 读失败轨迹、改 `guidance` / loop 参数，把 Qwen3.5-4B 在 ALFWorld 上从 held-in **0.359 → 0.703**、held-out **0.281 → 0.750**（≈翻倍，零过拟合）。
2. **kev 三轴先验注入**（本 repo 的增量贡献）：一个冻结的 System-1 读头在 **一次前向** 里给失败轨迹打出三轴（lens × lever × intent）概率向量，作为**软先验**喂给 meta-agent——held-out 上 ALFWorld **.844**（vs 纯 llm .781、硬 enforce .594）、WebShop **.630**（vs .560 / .567）。

**三轴对照结论（`llm` vs `prior` vs `enforce`）**：

| benchmark | baseline | llm | **prior** | enforce |
|---|---|---|---|---|
| ALFWorld（held-out pass） | .297 | .781 | **.844** | .594 |
| WebShop（held-out mean_reward） | .439 | .560 | **.630** | .567 |

> **软先验 > 硬裁决 > 不做先验**：kev 的弱读给强 meta 保留覆盖权就赢（prior）；做成绑定 gate 反而砍伤搜索空间（enforce）。

**与「训练 harness-agnostic 判别头」的区别**：kev 不是被进化的对象，也不被训练——它是进化循环外的一个**固定读头**，只在「读轨迹、归因失败模式」这段最便宜/最稳定/最抗偏置的环节替代 meta。本 repo 全部工作不碰任何权重。

---

## 2. 方法总览

### 2.1 只进化 harness（不碰权重）

每一轮（`run.py` 主循环）：

```
R0 裸基线 ──rollout 64 task──▶ 写 trajectories ──gate──▶ meta_agent.evolve() ──▶ R1 config
                                                                                  │
   R1 rollout ──gate──▶ evolve ──▶ R2 ...   （gate: mean_reward ≥ best − tolerance，否则回退 best）
```

- **进化载体是 `HarnessConfig` YAML**：只放两个 no-op processor——`SystemPromptProcessor`（系统提示）+ `TextEvolverSpecProcessor`（`guidance` 文本 + `history_n`/`loop_breaker_k`/`history_mode`/`history_obs_clip`/`tried_action_mask` 6 个 loop 标量）。meta-agent 把它当纯 YAML 读/改。
- **meta-agent = 强模型**读每轮轨迹、写下一轮 guidance。它发现的「任务语法」（ALFWorld 的精确动作词表、WebShop 的逐属性点击）是增益的真正来源——系统提示全程未动。
- **held-in 进化 / held-out 只测**：ALF 在 `eval_in_distribution`（140 可解局）进化 6 轮，`eval_out_of_distribution`（134）只测一次；WebShop 用官方 train 切片（`--start 1500`）进化、test 切片（`--start 0`）只测。held-in 涨而 held-out 不涨 = prompt 过拟合，两个都涨 = 真泛化。

### 2.2 kev 三轴先验注入（System-1 → System-2）

kev 是一个**冻结的判别式读头**：一次前向读出「失败机制」的概率向量，不生成、不训练。

**三轴词表**（benchmark-agnostic，定义在 `decision_prior.py`）：

| 轴 | 问题 | 取值 |
|---|---|---|
| **lens**（看什么） | 是否被 blocker 卡住 / 是否能力不足 | `lens_failure` / `lens_capability_gap` |
| **lever**（改什么） | 本轮失败落在哪些可核验动作模式 | **meta 诱导的 K 个二值谓词**（锚定动作签名，见 §2.3） |
| **intent**（意图） | candidate 会不会把失败翻正 / 把成功翻挂 | `corrective` / `preservative-transfer` / `preservative-lock` |

**两种注入模式**（`--decision-mode`）：

| 模式 | lens | lever | intent | 后端挂掉时 |
|---|---|---|---|---|
| `prior`（软先验） | 注入 brief | 注入 brief（noul，可覆盖） | 不启用 | warn + 回退 llm |
| `enforce`（硬 gate） | 注入 brief | 注入 brief（noul，可覆盖） | **candidate retrocheck**（`pass_prob<0.5` 拒绝该轮） | 硬 fail（loud abort） |

**数据流（System-1 → System-2）**：

```
失败轨迹（eval_passed == false）
        │  kev 一次前向（noul 判断，不生成）
        ▼
三轴概率向量 ──prior──▶ decision_priors.md ──注入──▶ meta brief（deepseek 可覆盖）
        │
        ├─ lens   失败机制：failure / capability_gap ──────────────┐
        ├─ lever  meta 诱导的 K 个二值谓词（动作签名）──────────────│ 注入 brief
        └─ intent 反事实：corrective / transfer / lock ──enforce──▶ candidate retrocheck
                                                                      （pass_prob < 0.5 → 拒绝该轮）
```

`prior` 只走注入 brief 一条路（lens + lever 都软注入、可被 meta 覆盖）；`enforce` 的 lever 同样软注入，但额外把 intent 做成 candidate 级反事实 gate（`pass_prob < 0.5` 拒绝该轮）。旧的「单 lever 绑定 gate」已退役——lever 现在是 soft noul 向量，不再做硬绑定。

### 2.3 lever 词表由 meta 诱导（`--decision-prior-source`）

lever 轴的词表**不是固定枚举**，而是每轮由 meta 模型从**本轮失败轨迹**里诱导出 K 个独立二值谓词——锚定在「机械可核验的动作签名」上（如 "never issued `take <target>`"、"`go to <X>` 连续 >3 次"）。只有锚定到轨迹文本（goal + 动作）上，kev 才判别得了；旧版抽象 4-way（`configuration/control/action/instruction`）让 kev 读头塌到 chance（置信 0.23）。

```
本轮失败轨迹（K 条紧凑 state）
   │  meta 读 → 提 K 个 {name, q, yes, no} 二值谓词     (induce_lever_predicates)
   ▼
专用词表 + 2 个固定 lens 题 ──全转成 JudgeQ──▶ kev 对每条轨迹打 noul 向量（K 个独立 p(true)）
```

| `--decision-prior-source` | 词表来源 |
|---|---|
| `meta`（默认） | 每轮由 meta 从本轮失败诱导（`induce_lever_predicates`） |
| `fixed` | 静态手写 `FIXED_PREDICATES`（ALF 5 个：`never_takes_object`/`many_invalid_actions`/`excessive_navigation`/`observation_loop`/`budget_exhausted`） |
| `none` | 无 lever 谓词，只剩 2 个固定 lens 题 |

> 两个正交开关别混：**词表谁提** = `--decision-prior-source`（meta/fixed/none）；**词表谁打分** = `--decision-backend`（`kev` 便宜读头 / `meta` 生成式 / `llm`=不建 prior）。你问的「meta 提词表 + kev 用」= `--decision-prior-source meta --decision-backend kev`，即两者默认组合。另有两个对照开关：`--decision-prior-constant F` 把每个先验概率抹平成 F（保结构、去数值），隔离「结构 vs 数值」的贡献。

> **为什么弱读反而赢**（机制，见 [REPORT §10](../../REPORT_visibility.md)）：meta 的瓶颈不是「能否理解」，而是「以什么认知成本/方差/偏置理解」。kev 判别式逐条读 logit 再聚合，天然把 base rate 数对、把失败模式抽象成可泛化的语义——soft prior 喂给强模型是正则化；硬 gate 越线让弱模型做合成级裁决，反而砍空间。

---

## 3. 结果速览

### 3.1 三轴对照（temp=0 确定性，64 task，6 轮 + held-out 探针）

**ALFWorld**（held-out = `eval_out_of_distribution` 探针，各臂 held-in best config 重放）：

| arm | held-in best（轮） | held-out | 泛化 Δ |
|---|---|---|---|
| baseline | .344（R0） | .297 | −.047 |
| llm | .734（R5） | .781 | +.047 |
| **prior** | .672（R5） | **.844** | **+.172** |
| enforce | .531（R3） | .594 | +.063 |

**WebShop**（held-out = 官方 test 切片）：

| arm | held-in best | held-out（pass / mean_reward） |
|---|---|---|
| baseline | — | .203 / .439 |
| llm | .281 / .536（R3） | .312 / .560 |
| **prior** | .312 / .598（R4） | **.344 / .630** |
| enforce | .250 / .562（R5） | .203 / .567 |

### 3.2 kev 先验的历史对照（Mode A，find/acquire 词表）

同一软先验、benchmark 专属词表时代（已被三轴取代）：ALFWorld held-out **llm .797 → kev .969**（+17pt）。

### 3.3 kev reviewer 的准确率锚点（A1/A2 gate GA）

kev 当「考官」判 ALF 失败轨迹的 `find/acquire`：准确率 **.672** vs 规则 tagger .352（**+0.32**），Cohen's κ 0.398。语义失败模式（找/取目标）是 kev 的增量所在；state-readable 机械模式（gridgames 掉坑/loop）上 kev 是**负增量**（规则 5 行 `has_loop` 完胜）。

### 3.4 规模扫描（2B / 0.8B 三轴）

同一三轴先验、只换 agent 底座（kev reviewer 不变）：

| cell | baseline | llm | prior | enforce |
|---|---|---|---|---|
| 2B ALF held-out | .047 | .141 | **.172** | .125 |
| 2B WS held-out（reward） | .483 | .504 | **.537** | .440 |
| 0.8B ALF | 全 0.000（二值地板，无失败模式可标注） | — | — | — |

**kev 相对增益随规模缩**：4B 上 prior−llm ≈ +.06~.07，2B 只剩 +.03，掉进 64-task 噪声。enforce 持续最弱。

> 纯 harness 进化（无 kev）的模型矩阵与 WebShop 迁移、gridgames decoding 根因，见 [EVOLUTION.md](EVOLUTION.md) 与根 [RESULTS.md](../../RESULTS.md)。注意：三轴对照用 **temp=0**（确定性）；纯 harness 进化的旧结果用 **temp=0.4**，两者绝对值不可直接比。

---

## 4. 模块清单

删减后的 live 树，一句话职责：

### 4.1 `recipe/agent_evolver/`（进化器本体）

| 文件 | 职责 |
|---|---|
| `run.py` | 进化器主循环：rollout + gate + evolve + journal + 三轴 decision 注入；argparse CLI |
| `spec.py` | `HarnessSpec` dataclass + 全部环境变量默认值 |
| `gate.py` | 回归门 `score_and_gate`（accept / revert 到 best） |
| `agents.py` | `<think>/<action>` 文本 ReAct rollout（被进化的东西） |
| `trajectories.py` | per-task `.md` 轨迹写入（frontmatter 带 `reward_info` / `eval_passed`） |
| `decision_prior.py` | 三轴词表（lens×lever×intent）+ ALF/WS 轨迹 parser/renderer + `DecisionPriors` |
| `candidate_retrocheck.py` | enforce 的 intent 轴：candidate 级反事实校验（`pass_prob<0.5` 拒绝） |
| `failure_labels.py` | ALFWorld `find/acquire/other` oracle 标签（读 `traj_data.json` 专家计划） |
| `webshop_labels.py` | WebShop 六路程序化标签（`no_search/not_bought/wrong_type/attr_mismatch/option_mismatch/price_over`） |
| `processors.py` | 循环 processor（`loop_breaker` / `history_format` / `tried_action_mask`） |
| `native/` | `HarnessConfig` 构造 + processor + prompt builder（`config.py`/`prompt.py`/`text_spec.py`） |
| `progress_diag/` | 进度诊断第一关（`oracle.py` 卡位 m + `judge.py` kev 卡位 k + `validate.py` 门） |
| `envs/textcraft/` `envs/sciworld/` | 两个新环境 wrapper + 失败标签（进程内 / HTTP server） |
| `skills/` | meta-agent 的 playbook skill（`alfworld-playbook` / `webshop-playbook`） |
| `multi_node/` | 跨节点：`run_multinode.sh` + `verify_meta_pod.sh` |

### 4.2 `harnessx/decision/`（离散决策接口，后端无关）

| 文件 | 职责 |
|---|---|
| `types.py` | `JudgeQ`（判断）/ `ChooseQ`（选项）/ `Decision` / 后端基类 |
| `kev.py` | `KevBackend`：kev 读头一次前向，noul 向量 + 概率 |
| `llm.py` | `LlmDecisionBackend`：meta-LLM 当 decision 后端（generative，逐轨迹一次调用） |
| `registry.py` | `get_decision_model`：按名构造后端 |

### 4.3 `harnessx/meta_harness/`（进化编排通用层）

| 文件 | 职责 |
|---|---|
| `agent.py` | `MetaAgent.evolve()`：读轨迹 → 写 config → journal |
| `decision_prior.py` | 三轴 pre-analysis 通用循环（benchmark 无关的部分） |
| `journal.py` | journal 读写 + 三轴 lever 合法性校验 |
| `replay.py` | `run_replay_gate_strict`：合成 smoke task 回放门 |
| `validate_workflow.py` | `EvolveValidator`：canonicalize → contract → diff → replay → novelty → evidence |

---

## 5. 安装与前置

### 5.1 环境

- **机器**：4× GPU（M402 80GB）。单机三轴 = 4 卡够；16 卡 cloudml 见 §6.4。
- **Python 3.12**（`uv` 管理）：

```bash
cd <REPO>                      # 仓库根（Harness-JEV）
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
# 内网环境需设 UV_INDEX_URL 指向内部 PyPI 镜像（URL 不写进本文档；见团队环境约定）
```

- **agent 模型**：`<MODEL_DIR>/Qwen3.5-4B`（vLLM serve，冻结；路径在 infra 脚本内配置）。
- **kev sidecar**：`<KEV_DIR>/kev`（代码）+ `<MODEL_DIR>/kev-4b`（训练产物 `head.pt`）。kev 的 base 模型 `Qwen3.5-4B-Base` 需 symlink 进 HF cache（infra 脚本里的 `kev_hf_cache()` 已处理）。
- **env server 依赖**：ALFWorld 用 conda env `ee-alfworld`、WebShop 用 `ee-webshop`（含 Java + spacy）、ScienceWorld 用 `ee-sciworld`（py3.8 + py4j），路径见 infra 脚本。
- **meta 后端**：deepseek-v4-pro（经 anthropic 兼容网关，key 走 `ANTHROPIC_API_KEY`）。多机 pod 够不到网关时自动回退 pod 挂载的 `/preset-models/public/DeepSeek-V4.1-Flash` 本地 vLLM（`multi_node/` 脚本已内置）。

### 5.2 前置探活

```bash
for p in 8200 8201 8202 8203; do curl -s --max-time 3 "http://127.0.0.1:$p/v1/models" >/dev/null && echo "vLLM :$p UP"; done
curl -s --max-time 3 "http://127.0.0.1:8090/v1/models" >/dev/null && echo "kev :8090 UP"
curl -s --max-time 3 "http://127.0.0.1:18082/health" >/dev/null && echo "ALF heldin :18082 UP"
```

---

## 6. 复现步骤

### 6.1 端口表

| 服务 | 端口 | 说明 |
|---|---|---|
| vLLM（agent Qwen3.5-4B） | :8200–8203 | gpu-mem 0.70（给 GPU0 上 kev 让显存） |
| kev sidecar | :8090 | 冻结读头，prior/enforce 臂都要 |
| ALFWorld held-in | :18082–18085 | `eval_in_distribution` |
| ALFWorld held-out | :18086–18089 | `eval_out_of_distribution`（探针期拉起） |
| WebShop | :18090 | 官方 full 集，12087 human goals，seed 233 |
| ScienceWorld | :18091 | py3.8，16-worker JVM 池 |
| TextCraft | 无（进程内） | 不占端口 |

### 6.2 一键：单机三轴对照（推荐）

```bash
cd <REPO>

# ① 起 infra（vLLM :8200-8203 + kev :8090 + env servers），bench = alfworld|webshop
bash start_three_axis_infra.sh alfworld

# ② 跑三臂 + held-out 探针（llm → prior → enforce 串行，逐臂 --decision-mode 不同）
#    one-arm 脚本已归档到 archive/scripts/three_axis_q0.sh；复活 = git mv 回根目录
setsid nohup bash three_axis_q0.sh alfworld >> /tmp/three_axis_q0.log 2>&1 &
```

`three_axis_q0.sh` 做三件事：① `llm` 臂（无 decision backend）→ ② `prior` 臂（`--decision-backend kev --decision-mode prior`）→ ③ `enforce` 臂（`--decision-mode enforce`），各 6 轮×64 task；最后 4 个 held-out 探针（baseline + 各臂 best config）并打印汇总。

### 6.3 手动 CLI（逐臂 / 探针）

```bash
export EVOLVER_AGENT_TEMPERATURE=0   # 确定性 rollout（三轴对照必须）

# llm 臂（无先验）
./.venv/bin/python -m recipe.agent_evolver.run alfworld --split heldin \
  --num-rounds 6 --num-tasks 64 --run-tag my_llm_heldin

# prior 臂（kev 软先验）
./.venv/bin/python -m recipe.agent_evolver.run alfworld --split heldin \
  --num-rounds 6 --num-tasks 64 --run-tag my_prior_heldin \
  --decision-backend kev --decision-mode prior --decision-base-url http://127.0.0.1:8090

# enforce 臂（kev 硬绑定 + candidate retrocheck）
./.venv/bin/python -m recipe.agent_evolver.run alfworld --split heldin \
  --num-rounds 6 --num-tasks 64 --run-tag my_enforce_heldin \
  --decision-backend kev --decision-mode enforce --decision-base-url http://127.0.0.1:8090

# held-out 探针（取某臂 best 轮 config 重放，只测不进化）
./.venv/bin/python -m recipe.agent_evolver.run alfworld --split heldout \
  --num-rounds 1 --num-tasks 64 --run-tag my_prior_heldout \
  --base-config recipe/agent_evolver/runs/evolve/my_prior_heldin/R5/config.yaml
```

WebShop 同构，只是：`webshop` 无 `--split`、held-in 用 `--start 1500`、held-out 用 `--start 0`、env 用 `--env-urls http://127.0.0.1:18090`。

### 6.4 多机（cloudml 16 卡）

cloudml/Volcano 提交一个 **2 pod × 8 GPU** 的 PyTorch job，`DOCKER_COMMAND` 填：

```bash
# BENCH=alfworld（默认）| webshop（env 或 $1）；RANK0 跑 llm→enforce、RANK1 跑 prior，共享盘 join
bash <REPO>/three_axis_cloudml.sh    # 已归档 → 先 git mv 回根目录
```

脚本自持 vLLM/kev/env 起停 + RANK 分派 + held-out 探针 + summary + 精确 PID teardown。meta 后端自动 detect pod 挂载的 preset 模型。提交前可用 `multi_node/verify_meta_pod.sh` 单 pod 探一下（① preset 挂载 ② vLLM dry-run ③ 网关可达性）。

### 6.5 四 benchmark 一键运行（`run_benchmarks.sh`，通用入口）

`run.py` 的 `benchmark` positional 直接支持四个环境，唯一差别是外部 env server。仓库根的 `run_benchmarks.sh` 把它包成参数化入口：**先幂等拉起 infra，再跑指定 benchmark**，token 走环境变量、不落任何文件。

| benchmark | env server | 说明 |
|---|---|---|
| textcraft | **无（进程内）** | 只靠 vLLM + meta，零 server |
| alfworld | :18082–85 heldin / :18086–89 heldout | 1 worker = 1 stateful shard，别撞端口 |
| webshop | :18090 | 官方 full 集（12087 goals）；held-in/out 用 `--start 0/1500` |
| sciworld | :18091 | py3.8 16-worker JVM 池 |

```bash
export ANTHROPIC_API_KEY='...'     # meta key：只从环境变量读，绝不写进脚本/日志/commit

bash run_benchmarks.sh textcraft                          # 零 server 最省事
bash run_benchmarks.sh alfworld --num-rounds 3 --num-tasks 64
bash run_benchmarks.sh webshop --decision-backend kev --decision-mode prior
bash run_benchmarks.sh sciworld --split heldout

# 四连跑（后台串行）
for b in textcraft alfworld webshop sciworld; do
  setsid nohup bash run_benchmarks.sh "$b" >> "/tmp/run_$b.log" 2>&1 &
done
```

- **infra 自动起**：脚本顶部调 `benchmarks/start_jev_infra.sh`（幂等，`/health` 探活、活着就 skip），覆盖 vLLM :8200–8203 + ALF held-in/out ×8 + WebShop :18090 + **ScienceWorld :18091**。`SKIP_INFRA=1` 可跳过。
- **token 是参数**：meta key 只从 `ANTHROPIC_API_KEY` 读（`EVOLVER_META_MODEL` / `EVOLVER_META_API_BASE` 可覆盖模型/网关）。刻意不搞 `--token` 命令行参数——那样会进 shell history / `ps`，不如 env var 干净。
- **`--decision-backend kev` 语义专属 ALFWorld**：kev 读头不迁移到 sciworld/textcraft 的动作词表，那两个用默认 `llm` 或 `meta`。

### 6.6 读结果

每个 run 目录（`recipe/agent_evolver/runs/evolve/<tag>/`）下：

- `comparison.json` —— 逐轮 `pass_rate` + `mean_reward`（**权威数字**，best = mean_reward 最高轮）。
- `R{n}/trajectories/{task_id}.md` —— 每 task 完整轨迹（frontmatter 带 `eval_passed`/`eval_score`）。
- `R{n}/evolve/decision_priors.md`（prior/enforce）—— kev 注入的失败机制先验。
- `R{n}/evolve/candidate_retrocheck.md`（enforce）—— intent 反事实校验。
- `learnings.md` —— meta-agent 的跨轮记忆。

> ⚠️ **别数中途日志的 win 数**：OK 任务快、FAIL 任务慢，中途数日志会虚高（一次 94% 实际 58.8%）。只看 `comparison.json`。

---

## 7. CLI 与环境变量参考

### 7.1 环境变量（`spec.py` 全部可覆写）

| 变量 | 默认 | 含义 |
|---|---|---|
| `EVOLVER_AGENT_MODEL` | `Qwen3.5-4B` | **agent 模型 id（切模型唯一开关；`--agent-model` flag 已删）** |
| `EVOLVER_AGENT_API_BASE` | `http://127.0.0.1:8200/v1` | agent vLLM base（单卡） |
| `EVOLVER_AGENT_API_BASES` | `8200..8203` 逗号串 | 多卡 vLLM（`run_benchmarks.sh` 拼好） |
| `EVOLVER_AGENT_TEMPERATURE` | `0.4` | 生成温度（**三轴对照设 0 做确定性**） |
| `EVOLVER_AGENT_TOP_P` | `0.8` | Qwen 推荐 top_p |
| `EVOLVER_AGENT_MAX_TOKENS` | `4096` | thinking on 需要大预算 |
| `EVOLVER_AGENT_ENABLE_THINKING` | `1` | S1 decoding：原生 thinking on |
| `EVOLVER_HISTORY_LENGTH` | `2` | 历史窗口（obs,action）对数 |
| `EVOLVER_ALF_MAX_STEPS` / `EVOLVER_WS_MAX_STEPS` / `EVOLVER_TEXTCRAFT_MAX_STEPS` / `EVOLVER_SCIWORLD_MAX_STEPS` | `50` / `15` / `30` / `40` | 各环境 step 上限 |
| `ANTHROPIC_API_KEY` | 无默认 | **meta-agent 的 key（token）**：只从环境变量读，绝不写进脚本/日志 |
| `EVOLVER_META_MODEL` | `anthropic/volcengine_maas/deepseek-v4-pro` | meta-agent 模型 |
| `EVOLVER_META_API_BASE` | 内网网关（回退 `ANTHROPIC_BASE_URL`） | meta 网关，URL 见 `run.py` 默认值 |
| `LITELLM_LOCAL_MODEL_COST_MAP` | 未设 | **必须在 `import litellm` 前设 `True`**（否则每进程 ~45s 冷启动） |

### 7.2 CLI 关键参数（`python -m recipe.agent_evolver.run --help`）

| 参数 | 说明 |
|---|---|
| `{alfworld,webshop,textcraft,sciworld}` | benchmark（positional） |
| `--split heldin\|heldout\|compat100` | ALF/textcraft/sciworld；WS 忽略（用 `--start` 切） |
| `--num-rounds N` | 总轮数（R0..R{N-1}；R0=裸基线） |
| `--num-tasks N` / `--start K` / `--seed S` | 采样：数量 / 起始 task_id / seed（ALF 默认 1234，WS 默认 0） |
| `--decision-backend {llm,kev,meta}` | `llm`=关；`kev`=注入 System-1 先验（ALF/WS）；`meta`=meta-LLM 自己答三轴题（新环境用） |
| `--decision-base-url URL` | kev sidecar 地址（默认 :8090） |
| `--decision-mode {prior,enforce}` | 软先验 / 硬绑定（intent candidate retrocheck） |
| `--decision-prior-source {meta,fixed,none}` | lever 词表来源消融：每轮诱导 / 手写固定 / 只 lens |
| `--decision-prior-constant F` | 对照实验：把每个先验概率抹平成 F（保结构、去数值）；`None`=用真值 |
| `--task-fliprate-path PATH` | 预计算 per-task 历史翻转率表（默认 `data/predicted_affected_ablation/task_fliprate.json`） |
| `--gate-task-weight {uniform,fliprate}` | gate 的 `round_reward` 算法：plain mean / `w=1-flip_rate` 加权 |
| `--base-config YAML` | 探针用：从外部 config 起跑（不进化） |
| `--agent-api-bases` / `--env-urls` | 多卡 vLLM / env server 分片端点（逗号分隔） |
| `--meta-model` / `--meta-api-base` | meta 后端覆盖 |
| `--evolve-cost` / `--evolve-steps` / `--evolve-wall-clock` | evolve 预算：美元 / 步数 / 墙钟 |
| `--regression-tolerance` / `--cost-weight` | 回归门容差 / 成本权重 |
| `--concurrency` | 最大并发 task（默认 = len(agent-api-bases)） |
| `--run-tag TAG` | 输出目录名 |

### 7.3 按优先级分档

日常跑只记**第一档**；做消融才动**第二档**；**第三档**基本由 `run_benchmarks.sh` / infra 脚本代管。

**第一档（决定结果）**

- **三轴先验开关**：`--decision-backend {llm,kev,meta}` + `--decision-mode {prior,enforce}` + `--decision-prior-source {meta,fixed,none}`
  - 核心组合「kev 软先验」= `--decision-backend kev --decision-mode prior`（`prior-source` 默认 `meta`，不用写）
  - 纯 llm 对照臂 = 全默认（backend `llm` 时 mode/prior-source 都是空转）
- **进化/测试切分**：`--split {heldin,heldout,compat100}`（WS 忽略，改用 `--start 0/1500`）
- **规模/轮数/种子**：`--num-tasks` / `--num-rounds` / `--seed`（冒烟试 `--num-rounds 1`）
- **held-out 探针**：`--base-config recipe/agent_evolver/runs/evolve/<tag>/R5/config.yaml`（只测不进化）

**第二档（对照/调优）**

- `--decision-prior-constant F` —— 抹平先验数值、保结构（隔离「结构 vs 数值」贡献）
- `--gate-task-weight fliprate` + `--task-fliprate-path` —— gate 按历史翻转率加权（`w=1−flip_rate`）降噪
- `--regression-tolerance` —— gate 容差（`mean_reward ≥ best − tol` 才 accept）
- `--concurrency` —— 最大并发（默认已按 `len(agent-api-bases)` 拉满）

**第三档（运维/预算）**

- `--agent-api-bases` / `--env-urls` —— 多卡 vLLM / env server 分片端点
- `--run-tag` —— 输出目录名
- `--meta-model` / `--meta-api-base` —— meta 后端覆盖
- `--evolve-cost` / `--evolve-steps` / `--evolve-wall-clock` / `--cost-weight` —— 进化预算

---

## 8. 四环境接入

| 环境 | 接入方式 | 隔离 | 关键坑 |
|---|---|---|---|
| **TextCraft** | 进程内 `TextCraftEnvWrapper`（import vendor，无 env server） | 单进程内复用 | 只靠 vLLM + meta，零 server |
| **ALFWorld** | HTTP env server ×8 | **1 worker = 1 stateful shard**（`env_urls[i % len]`） | 两个 run 撞同一端口 → `RemoteProtocolError` |
| **WebShop** | HTTP env server ×1 | ThreadingHTTPServer 线程安全，每次 `/create` 独立 uuid | 官方 full 集；held-in/out 用 `--start` 切 |
| **ScienceWorld** | HTTP env server ×1 + 16-worker JVM 池 | py4j 靠 `$JAVA_HOME` 找 java | worker `busy` 只在 `/destroy` 释放，被杀 run 会泄漏 |

详细接入 SPEC：TextCraft 见 [TEXTCRAFT_SPEC.md](TEXTCRAFT_SPEC.md)，ScienceWorld 见 [SCIWORLD_SPEC.md](SCIWORLD_SPEC.md)。

---

## 9. 诊断与失败标签

### 9.1 程序化失败标签

- **ALFWorld**（`failure_labels.py`）：从专家计划（`traj_data.json` 的 `plan.high_pddl`）读 target / 需求动作，判 `find`（没 take 到 target）/ `acquire`（take 到但没完成下一步）/ `other`。
- **WebShop**（`webshop_labels.py`）：六路标签 `no_search` / `not_bought` / `wrong_type` / `attr_mismatch` / `option_mismatch` / `price_over`，由 env 在 buy 时返回的 `reward_info`（`r_type`/`r_att`/`r_option`/`r_price`）分解得到。`reward>=0.999` 是成功、不打标签。

### 9.2 进度诊断第一关（progress_diag）

「kev 判卡位 k vs oracle 卡位 m」的对齐门：

- **ALFWorld = GO**：oracle 卡位（find/acquire 是动作存在性）与 kev 判卡位**同构可读**——oracle 从专家计划推 m，kev 读轨迹推 k，卡位准确率 0.905 vs 常数基线 0.714。
- **WebShop = NO-GO（方法边界）**：oracle 依赖 agent **不可见**的结构化字段（`r_type`/`r_att`/`r_option`/`r_price`），在「kev/meta 只能看 agent 可见信息（goal、动作、商品详情页）」的无泄漏约束下是**信息论不可判定**的。因此 WS 第一关只写方法边界（`progress_diag/webshop_oracle.md`），**不留代码**。

> 结论：第一关在 ALFWorld 上通过；WebShop 的「值匹配型」oracle 因信息边界弃用。详见 `progress_diag/align.md`。

---

## 10. 安全红线（约定，非真实凭证）

本节写的是**团队约定**，不包含任何真实 token/key/URL/路径：

1. **token 只从环境变量读**：`ANTHROPIC_API_KEY` 等 key 一律走 `os.environ.get(...)`，代码里不写死、命令行不加 `--token` 参数（避免进 shell history / `ps`）。
2. **日志/报告打码**：任何日志、`REPORT*`、`learnings.md`、commit message 都不出现 key/token/secret 字面量。
3. **内部 URL / 主机名不落 README**：网关 URL、共享盘绝对路径只在 infra 脚本里配置，文档用 `<REPO>` / `<MODEL_DIR>` 占位或「见脚本」。
4. **删改文件不碰凭证**：精简/归档只动结构与死代码，不动任何 `api_key=` / `token=` 赋值行。
5. **提交前 secret 扫描**：跑 `data/secret_scan.py` + `grep -rnE 'key|token|secret'` 复核，命中即回退。
6. **README 自身只写约定**：本文件不出现真实凭证、内网地址。

---

## 11. 坑

- **切模型只认 `EVOLVER_AGENT_MODEL`**：`--agent-model` flag 已删；且 vLLM 的 `--served-model-name` 必须与之相等。
- **temp 必须是 0**：三轴对照逐 seed 比较时，vLLM bf16/重启噪声会污染 ±2–4pt；只有 temp=0 贪心才在同 run 内可比（跨 run 绝对值仍要打折）。
- **ALF env 是 1 worker = 1 env server 的 stateful shard**（`env_urls[i % len]`）：**别让两个 run 同时占同一 held-in/held-out 端口**，否则并发打同一 server → `RemoteProtocolError`（STEP-CONN-ERROR 记 task 失败）。
- **enforce 的硬 fail**：`--decision-mode enforce` 且后端挂 → loud abort（不静默回退），这是设计（防止硬 gate 假装通过）。
- **meta 后端**：deepseek fork 无 `temperature` 参数、`max_tokens ≥ 4096`（否则 thinking 吃光预算）；`LITELLM_LOCAL_MODEL_COST_MAP=True` 必须在 `import litellm` 前设（否则每进程 ~45s）。
- **infra 会被清**：机器重启 / supervisord 周期清掉 vLLM 和 env server，报 `httpx.ConnectError` 先重跑 `start_three_axis_infra.sh`（或 `run_benchmarks.sh` 会自动拉起）。
- **`--agent-api-bases` 是逗号分隔**（`split(",")`），空格分隔会整串当 api_base → 404 `Not Found`。
- **ScienceWorld env server 无超时回收**：worker `busy` 只在 `/destroy` 释放，被杀 run 的 in-flight 任务会永久泄漏（`free` 一路掉）。长跑前先重启 env server 清到 free:16。
- **ScienceWorld gold planner 非确定**：`task_selection.json` / `splits.json` 是**锁死工件**（sha256 见 `envs/sciworld/SCIWORLD_BASELINE.md`），下游全用这个固定 split。

---

## 12. 相关文档索引

| 文档 | 内容 |
|---|---|
| [EVOLUTION.md](EVOLUTION.md) | 纯 harness 进化：数据流图、可进化面、模型矩阵、WebShop 迁移、gridgames decoding |
| [KEV_THREE_AXIS.md](KEV_THREE_AXIS.md) | 三轴词表定义、prior/enforce 实现细节、改动文件清单 |
| [META_AGENT_DECISIONS.md](META_AGENT_DECISIONS.md) | meta-LLM 的决断清单 + JEV 可替代性分析 |
| [A1A2_RESULTS.md](A1A2_RESULTS.md) | kev reviewer 的 A1/A2 gate GA 结果 |
| [TEXTCRAFT_SPEC.md](TEXTCRAFT_SPEC.md) / [SCIWORLD_SPEC.md](SCIWORLD_SPEC.md) | 两个新环境接入 SPEC（任务选择/切分/env 语义/标签/门） |
| [progress_diag/align.md](progress_diag/align.md) | 进度诊断第一关指标定义 + GO/NO-GO |
| [../../RESULTS.md](../../RESULTS.md) | 全结果汇总（ALF/WS/FL/SK + 规模矩阵 + 三轴 + 系统-1 参照系） |
| [../../REPORT_visibility.md](../../REPORT_visibility.md) | visibility 报告（S2 基座 + A1/A2 + B1 + 三轴 + P1-P9 判定 + 机制讨论） |
