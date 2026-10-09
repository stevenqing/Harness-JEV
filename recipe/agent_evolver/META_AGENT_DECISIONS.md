# Harness-JEV：meta-evolve 里 LLM 的决断清单 + JEV 可替代性

> 目标模型 Qwen3.5-4B · meta 模型 deepseek-v4-pro · 只进化 harness（权重永不训练）。
> 本文回答一个问题：`MetaAgent.evolve()` 里，meta-agent 的 LLM 到底做了哪些决断、每个决断落在哪个 module、哪些决断能被 JEV（确定性代码）替代。
> 配套阅读：`EVOLUTION.md`（方法与结果）、记忆 [[meta-actionspace-isolation]]。

---

## 0. 结论先行

meta-agent 的 LLM 在 evolve 里做的决断是一条「**读 → 诊断 → 选杠杆 → 写 → 记 → 自检**」的决策链。每个决断都同时落在两个 module 之间：

- **指令载体** —— `SOUL.md` / 某个 skill 教它「怎么决」；
- **约束载体** —— processor / 后置闸门限它「不能怎么决」。

按「JEV 能否替代」分三档：

| 档 | 内容 | JEV 替代性 |
|---|---|---|
| **① 执行/记录/校验层** | gate / novelty / evidence / changeset / canonicalize / replay / journal 回填 | **已决定论化**，零 LLM，本来就是 JEV/harnessx |
| **② 诊断-选杠杆层** | 聚类门槛、lever 选择、证据筛选、failure-mode 分桶 | **可封顶式替代**（固定「失败模式→杠杆」策略表），正是 pre-meta 手写 lever 血统，但封顶在人类想象力 |
| **③ LLM 本质层** | 开放式因果假设、文本/代码生成、整体 Pareto 权衡、反事实校验 | **替代即失去价值**：这是 meta-agent 唯一不可替代增量 |

---

## 一、完整扫描：LLM 在 evolve 里做的 14 个决断

| # | LLM 做的决断 | 指令载体（谁教它） | 约束载体（谁限它） | JEV 能否替代 |
|---|---|---|---|---|
| 1 | **读哪些轨迹 / 扫多深**（证据筛选，tool use） | `SOUL.md` Loop 步3 + `analyze`「Read strategy」 | Glob/Grep/Read/`spawn_reflect_worker`；token=`CompactionProcessor`(200k)、步数=`StepDeadlineReminder`(260)、钱=`CostGuard` | **部分**：JEV 能确定性挑「最低 reward / 最长 steps / 指定失败桶」轨迹定向注入；「读到哪段 body」仍需 LLM |
| 2 | **诊断模式 + 归因**（lens×lever 三轴：Failure/Capability/Success × Config/Control/Action/Instruction） | `analyze`「two axes of reflection」 | — | **部分**：JEV 能做确定性 failure-mode 分桶（ALF find/acquire、WS search/click）；「因果假设」难 |
| 3 | **聚类门槛**（systemic vs idiosyncratic，≥2 task 同根因才算） | `analyze`「Systemic vs idiosyncratic」 | `candidates.md`「Tasks affected」字段 | **能**：字符串/特征匹配即可判「是否 ≥2 task 同 shape」 |
| 4 | **反事实校验**（retroactive check Variant A/B/C：这个 fix 早在了会不会过） | `analyze`「retroactive check」 | — | **难**：需 counterfactual 推理，JEV 只能做弱代理 |
| 5 | **选哪个杠杆 + 排除相邻杠杆**（"Why X not Y"） | `analyze`「Disambiguating adjacent levers」 | `candidates.md`「Why X not Y」必填字段 | **能封顶式替代**：正是 pre-meta 手写 lever 表（+P/+U/+O/+K）干的事 |
| 6 | **作者化**：写 guidance 文本 / 模板 / `@tool` / processor / 调 knob | `reference`（四种杠杆的机械写法） | Write/Edit/Bash；`literals` 扫描 | **本质 LLM**：文本/代码生成，JEV 只能模板化 |
| 7 | **起草 candidate**（header `## Candidate C-N` + 全字段） | `analyze`「candidates.md」 | 后置 evidence 门解析 header 正则 | **部分**：格式强约束（机械），内容靠 LLM |
| 8 | **ship 改动 vs 显式 no-op** | `SOUL.md` hard invariant 1 + `TASK.md`「Decision contract」 | 后置「missing config」门 | **能**：二值决策，下游已决定论化 |
| 9 | **全局优先 / Pareto 排序**（global gain > regression risk > cost shift） | `SOUL.md`「Pareto thinking rule」+ `analyze`「Candidate ordering」 | — | **部分**：可用启发式代理，整体 counterfactual 难 |
| 10 | **能力 gap 三分类**（harness 缺陷→修；模型能力 gap→跳过 + 记一行） | `SOUL.md`「Evolution philosophy」 | — | **部分**：「模型不会 vs harness 没给」需领域判断 |
| 11 | **写 journal 记一轮**（memory：frontmatter + prose） | `journal` skill（schema + 必填键） | journal 解析器只认 frontmatter | **能**：JEV 已回填 `gating_outcome`/`attribution`；正文可模板化 |
| 12 | **自检再交**（跑 canonicalize/dry_fire/contract/literals） | `validate` skill | `EvolveValidator` 后置五道闸 | **能（且冗余）**：post-flight 本来就会跑，LLM 自检只是省一轮成本 |
| 13 | **预算/步数怎么花**（tool use 分配） | — | CostGuard/StepDeadlineReminder/LoopDetection 硬限 | **部分**：上限是确定性的，「怎么花」是 LLM |
| 14 | **是否参考 playbook**（读 benchmark 攻略 + 应用失败模式→杠杆映射） | `<bench>-playbook`（JEV 写的） | — | **已经是 JEV**：playbook 内容就是 JEV 手写的固定策略表 |

---

## 二、三档边界（展开）

### ① 已经被 JEV / harnessx 决定论化（LLM 其实没在做，或在做重复功）

- 决断 8 的下游、12、11 的回填，以及 post-flight 五道闸（canonicalize→novelty→evidence→changeset→replay）—— 这些是 `EvolveValidator` + `compute_changeset` + `journal.py`，**零 LLM**。
- LLM 的「自检」（12）只是把这些闸在 `end_turn` 前提前跑一遍省成本，逻辑上可完全交给 JEV。
- 决断 14（playbook）——内容本来就是 JEV 写的，已经是固定策略表，不是 LLM 现发现。

### ② 能被 JEV 用固定策略表替代，但会「封顶」

- 决断 3（聚类）、5（选杠杆）、1 的筛选部分、2 的 failure-mode 分桶 —— 这些正是 **pre-meta 的 HarnessX-jev 手写 lever 血统**（+P 证据注入 / +U 未去过容器 / +O opened / +K deliver 锚…）在做的事：`失败模式 → 固定杠杆映射`。
- **关键 caveat**：这样替代的产物是「人类已经想到过的杠杆」。而记忆里 meta-agent 的两个最大赢面（ALFWorld 动作词表 +33pt、WebShop click 每个属性选项 +20pt mean）**恰好是手写表里没有的「benchmark 评分语法」**。所以 ② 替代得了动作，替代不了「发现」。

### ③ LLM 本质，JEV 替代即失去价值

- 决断 2 的**开放式因果假设**、6（文本/代码生成）、9 的整体 Pareto 权衡、4 的反事实校验。
- 这些是 meta-agent 相对手写进化器的唯一不可替代增量：**从轨迹里摸出 benchmark 的隐含评分/动作语法**，然后写成 guidance 给弱模型照做。JEV 的代码没有「没想到过」这个状态，所以顶不到这里。

---

## 三、一句话结论

> **LLM 在 evolve 里做的 14 个决断，落在「SOUL.md Loop + 4 个 skill + playbook」教它、又被「9 个 processor + 5 道 post-flight 闸」限它的夹层里。其中执行/记录/校验层（①）已经 100% 是 JEV；诊断-选杠杆层（②）可以用 JEV 的手写 lever 表替代、但那是 pre-meta 的老路且封顶在人类想象力；真正不可替代的是③——「从失败轨迹里发现 benchmark 的评分语法」这一跳，它才是这次进化 held-out 翻倍的来源。**

**若要让 JEV 更激进接管**：不要碰③；把①里 LLM 的重复功（自检、journal 正文、证据筛选）彻底移到 JEV，把 LLM 的算力预算全部留给③。

---

## 附：module 速查（各载体在决策链中的角色）

| module | 角色 | 位置 |
|---|---|---|
| `SOUL.md` | 7 步 Loop + Pareto 规则 + Evolution philosophy + 硬不变量（ship/no-op、只写 output_dir、绝对 file://、replay） | `harnessx/meta_harness/workspace/SOUL.md` |
| `analyze` skill | 三轴诊断、聚类门槛、反事实校验、lever 消歧、candidates.md schema | `.../workspace/skills/analyze/SKILL.md` |
| `journal` skill | 跨轮 memory schema + 必填 frontmatter（`round/hypothesis_id/levers/predicted_affected/cited_candidates/...`） | `.../workspace/skills/journal/SKILL.md` |
| `reference` skill | 四种杠杆（@tool / MultiHookProcessor / template / knob）的机械写法 | `.../workspace/skills/reference/SKILL.md` |
| `validate` skill | 自检 CLI（canonicalize/dry_fire/contract/literals） | `.../workspace/skills/validate/SKILL.md` |
| `<bench>-playbook` | benchmark 失败模式 + lever 映射 + 红线（JEV 手写） | `recipe/agent_evolver/skills/*-playbook/SKILL.md` |
| `build_meta_agent_harness_config` | meta-agent 的工具集 + processor 约束链 | `harnessx/meta_harness/agent.py:98` |
| `_render_task_brief` | TASK.md 任务书 + Decision contract | `harnessx/meta_harness/agent.py:719` |
| `compute_changeset` | 两份 canonical config 浅 diff | `harnessx/meta_harness/agent.py:439` |
| `EvolveValidator` | post-flight 三阶段（validity/policy/advisory）五道闸 | `harnessx/meta_harness/validate_workflow.py` |
| `journal.py` | 归因回填（`compute_attribution`/`fill_gating`/`build_context`） | `harnessx/meta_harness/journal.py` |
