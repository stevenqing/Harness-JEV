# 分层定义 — harness config 字段到四层（`recipe/agent_evolver/layers.py`）

对应 `specs/kev_ranker_step0c.md` §2。两个函数：`layers_of(parent, candidate) -> dict[layer, chars]` 和 `project(parent, candidate, layer) -> config`。四层与 `decision_prior.py::AXIS_QUESTIONS` 里 lever 的四个选项一一对应。

## 字段到层映射（判定优先级从上到下）

| 层 | 字段路径规则 | 理由 |
|---|---|---|
| `action` | 路径含 `tools` / `tool_registry` / `actions` | 动作空间 / 工具面的改动（当前 carrier config 里尚无） |
| `control` | 改动字段位于 `processors[]` 条目内，且该条目的 `_target_` 引用 `processors.control.` | 机械式后处理 hook |
| `instruction` | 路径含 `prompt` 或 `guidance` | 自由文本指令面 — `SystemPromptProcessor.system_builder.prompt` 与 `TextEvolverSpecProcessor.guidance` |
| `configuration` | 其余所有改动字段 | 标量循环参数（`history_n`、`history_mode`、`history_obs_clip`、`loop_breaker_k`、`tried_action_mask`）+ 顶层接线/基础设施 |

`tried_action_mask` / `loop_breaker_k` 的说明：它们在语义上是机械式干预，但在此 carrier config 里是现有 `TextEvolverSpecProcessor` 上的标量 kwargs，按参考 skill 归为 Configuration lever（「就地调现有 processor 的 kwargs」），故在此归 `configuration`。若未来某对把 `harnessx.processors.control.*` processor 换进来，那部分归 `control`。

## 改动大小

- 两边都是字符串 → `max(len(old), len(new))`
- 增/删（一侧为 `None`）→ `len(str(存在侧))`
- 标量 → `len(str(改动后的值))`
- 主层 = `argmax(size)`；触碰 >1 层的对记作 multi-layer

## project 语义

`project(parent, candidate, layer)` 确定性投影：逐叶遍历，只保留归类进 `layer` 的改动，其余叶回退到 parent 的值。

- 修改（old→new）：`layer` 匹配 → new；否则 → old
- 新增（仅 candidate 有）：`layer` 匹配 → new；否则丢弃
- 删除（仅 parent 有）：`layer` 匹配 → 删除；否则恢复 parent 值

投影后 `layers_of(parent, project(parent, candidate, layer))` 只在该 `layer` 非零（G1 的断言）。
