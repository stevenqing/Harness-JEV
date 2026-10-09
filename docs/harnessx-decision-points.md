# Harness-JEV：harnessx 全组件决策点 × jevlike model 可替代清单

> 目标：把 harnessx 9 个 module 里**所有决策点**翻出来，标成 **判断 / 选项** 两类，落成一张能照着逐个替换的工作清单。
> 底座复用 Qwen3.5-4B-Base（冻结）+ 轻量 LoRA/pointer head（kev 式），head 维数 = 该决策的候选集大小。
>
> 配套：[[harnessx-jev-strong-meta-backend]]、[[meta-actionspace-isolation]]、
> `recipe/agent_evolver/META_AGENT_DECISIONS.md`（meta-agent 单点的 14 决断清单）。

---

## 0. 判定口径（先定死）

1. **决策点按「现在谁在做」分三源**，三源**都算决策、都可被 jevlike 离散头替换**：
   - **LLM**：贵、每回合烧一次大模型 → 替换动机是**省成本/省延迟**；
   - **规则**：脆、手写 fingerprint/阈值/取最旧 → 替换动机是**从手写升级到可学习**；
   - **embedding**：固定、任务无关的相似度 → 替换动机是**从固定升级到任务相关**。
2. **形态只有两种**，没有连续打分：
   - **判断**：给输入打一个闭集标签（过/不过、对/错、是/否、loop/no-loop、worth/not）→ 2~3 类分类头。
   - **选项**：从闭集候选里选一个（选哪个动作/哪个枚举值/哪条记忆）→ 候选数维 pointer head。
3. **连续值不出现**：gate 的 `mean_reward` 是 env 确定性值；PRM 若要进，离散成「好步/坏步」。
4. **判别线 = 语义判断 vs 纯阈值**：只有「在做语义判断」的规则/embedding 才值得升级成模型；
   纯算术/预算检查（token 数、成本上限）无语义可学，留阈值即可。

## 0.1 结论先行（读代码后的真实数字）

**~10 个决策点，全判断/选项型**：
- **4 个 LLM 决策点**（替换 = 省钱省延迟）；
- **~6 个规则/embedding 决策点**（替换 = 升级，从手写 fingerprint / oldest-n / 固定 embedding 到可学习）。

「选项」型只出现在 rollout 动作选择（主模型 gen）和 tool 枚举消歧，其余全是「判断」。

---

## 1. 按 9 module 分类

### 1. MODEL —— 模型选择 / 路由

| 组件 | 决策 | 类型 | 现在谁做 | head | 输入 | 动机 |
|---|---|---|---|---|---|---|
| `multi_model/model_router.py::ModelRouterProcessor` | 查询复杂度 → 路由 small/main | **判断** | **LLM**：small 子 harness 跑 `_ROUTER_PROMPT`，产出 `{"complexity":"simple\|complex","confidence":...}` | 2（simple/complex） | task_description | 省成本：砍掉每次任务启动的一次模型调用 |

### 2. CONTEXT —— 上下文组装

- `system_prompt.py` / `user_wrapper.py` / `env_context_injector.py`：模板/规则，无运行时语义决策。
- 「历史截断留哪段」= 窗口（确定）＋ compaction（在 CONTROL）。
- **结论：无语义决策点。** 未来若加「留/丢哪段」的判断，才成为可替点。

### 3. MEMORY —— 记忆管理

| 组件 | 决策 | 类型 | 现在谁做 | head | 输入 | 动机 |
|---|---|---|---|---|---|---|
| `memory/memory_retrieval.py` | 哪条记忆相关 | **选项**（或逐条判断） | **embedding**：`memory.retrieve(query, top_k)` 固定相似度 | top-k pointer（或每候选 2 类相关/不相关） | query + 候选记忆 | 升级：固定 embedding 不知道任务相关性 |
| `memory/memory_extraction.py` | 哪段值得记 | **判断** | **规则**：`OldestMessagesExtractor` 取最旧 n 条 | 2（worth/not） | 逐条消息 | 升级：oldest-n 是任意规则，学不到 salience |

### 4. TOOLS —— 工具生态

- 工具过滤（哪些可见）：静态配置/规则，无运行时语义决策。
- 「选哪个工具调用」：主模型 gen 的职责（快慢系统慢侧），非 harness 处理器决策。
- **结论：无独立 harness 语义决策点。**

### 5. SANDBOX —— 执行环境

- 隔离模式（Local/Docker/E2B）= 静态配置。**结论：无。**

### 6. EVALUATE —— 评估 / 奖励

| 组件 | 决策 | 类型 | 现在谁做 | head | 输入 | 动机 |
|---|---|---|---|---|---|---|
| `evaluation/llm_judge.py::LLMJudgeProcessor` | 答案过不过（5 字段 verdict，核心 pass/fail） | **判断** | **LLM**：judge 子 harness 生成 JSON verdict | 2~3（pass/fail/partial） | Q + A + rubric + trajectory summary | 省成本；但 `signals`/`missing_capability` 是生成型，留 LLM |

### 7. CONTROL —— 控制 / 安全（13 个 processor）

| 组件 | 决策 | 类型 | 现在谁做 | head | 输入 | 动机 |
|---|---|---|---|---|---|---|
| `self_verify.py::SelfVerifyProcessor` | 自检对不对 | **判断** | **LLM**（⚠️ 不是独立 judge——是给主模型注入验证消息**多跑一步**） | 2（对/错） | task + 我的产出(+测试结果) | 省延迟：省掉那一步主模型往返 |
| `sycophancy_detector.py::SycophancyDetector` | 是否谄媚 | **判断** | **LLM**（regex 命中 → 可选 judge 确认 + chat/task 分类） | 2（+chat/task 双标） | 最近几轮 | 省成本 + 把可选 judge 固化 |
| `loop_detection.py::LoopDetectionProcessor` | 是否绕圈 | **判断** | **规则**：`_compute_fingerprint` + `_consecutive_tail` 阈值 | 2（loop/no-loop） | 最近动作/tool_call 历史 | **升级：手写 fingerprint 只认字面重复，学不到语义 loop** |
| `compaction.py::CompactionProcessor` | 留/丢哪段 | **判断** | **规则**：`_default_summarize` 截断尾部 N 字符 | 2（keep/drop，逐段） | 逐段消息 | 升级：纯长度截断不看内容；但「总结」本身是生成型，留 LLM |
| `tool_call_correction.py::ToolCallCorrectionLayer` | 哪个枚举值 | **选项** | **规则**：`_fold_enum_case` 大小写折叠 | enum 候选数 | 坏值 + param schema | 升级（弱）：大小写折叠学不到语义消歧 |
| `todo_check.py` | 剩余 todo 是否阻塞 | **判断**（弱） | **规则**：读 `todo_write` 快照的 status 字段 | 2（阻塞/不阻塞） | todo 列表 | 弱：现状只是查字段，真正的「做完没」是潜在判断 |
| `cost_guard.py` / `token_budget.py` / `parse_retry.py` / `repeated_edit_detector.py` / `tool_failure_guard.py` / `bg_install_guard.py` / `rl_signal.py` | 阈值/标志判断 | — | **规则**：硬阈值 | — | — | **不升级**：纯算术/预算，无语义可学 |

### 8. OBSERVE —— 可观测

- journal / OTel / checkpoint / recovery = 记录，无决策。**结论：无。**

### 9. TRAIN —— 训练桥

- 轨迹 → SFT/RL 标注 = 规则。
- meta-agent evolve = 14 决断（见 `META_AGENT_DECISIONS.md`）：
  - ③ 层（开放式因果假设 / 文本生成 / Pareto 权衡）→ **不可替**；
  - ② 层（诊断 / 选杠杆）→ 可封顶式判断/选项，但等于退回手写 lever 表，丢「发现 benchmark 评分语法」的增量。
- **结论：③ 层不替；② 层可替但价值存疑。**

---

## 2. 汇总

### 2.1 LLM 决策点（替换 = 省成本/省延迟）

| # | Module | 组件 | 决策 | 类型 | head |
|---|---|---|---|---|---|
| 1 | 1 MODEL | `model_router` | 复杂度 → 路由 | 判断 | 2 |
| 2 | 6 EVALUATE | `llm_judge` | 过不过 | 判断 | 2~3 |
| 3 | 7 CONTROL | `self_verify` | 对不对 | 判断 | 2 |
| 4 | 7 CONTROL | `sycophancy_detector` | 谄媚? | 判断 | 2(+2) |

### 2.2 规则 / embedding 决策点（替换 = 升级到可学习）

| # | Module | 组件 | 决策 | 类型 | head | 升级点 |
|---|---|---|---|---|---|---|
| 5 | 7 CONTROL | `loop_detection` | 是否绕圈 | 判断 | 2 | 手写 fingerprint → 语义 loop |
| 6 | 3 MEMORY | `memory_retrieval` | 哪条记忆相关 | 选项 | top-k pointer | 固定 embedding → 任务相关 |
| 7 | 3 MEMORY | `memory_extraction` | 哪段值得记 | 判断 | 2 | oldest-n → salience |
| 8 | 7 CONTROL | `compaction` keep/drop | 留/丢哪段 | 判断 | 2 | 长度截断 → 内容相关 |
| 9 | 7 CONTROL | `tool_call_correction` enum | 哪个枚举值 | 选项 | enum 数 | 大小写折叠 → 语义消歧 |
| 10 | 7 CONTROL | `todo_check` | 阻塞? | 判断（弱） | 2 | 查字段 → 真判断 |

（rollout 动作选择 = 主模型 gen 的「选项」型，最大速度杠杆，属快慢系统慢侧，单列。）

## 3. 一句话结论

> harnessx 的决策面 = **~10 个判断/选项型决策点**，现在由 **LLM（4）/ 规则（~5）/ embedding（1）** 三源实现，
> 三源都可被同一个 jevlike 离散头（冻结 4B-Base + LoRA/pointer head，head 维数=候选集大小）替换——
> LLM 换 = 省钱省延迟；规则/embedding 换 = 从手写 fingerprint、oldest-n、固定相似度**升级到可学习**。
> 判别线是「语义判断 vs 纯阈值」：做语义判断的（loop/相关/salience/keep-drop）值得学，纯预算检查（token/成本）留阈值。
> 最终所有决策收敛成一个可进化原语：**learned discrete head**——这才是 harness 真正能「进化」的最小单位。
