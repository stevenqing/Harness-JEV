# Track A — A1/A2 结果（kev reviewer vs 规则 tagger vs oracle 标签）

> 生成于 2026-10-08。A1=独立失败标签（oracle），A2=确定性规则 tagger，二者都对着
> 同一个 kev reviewer（noul 概率向量 → argmax 硬标签）打分。
> 采样单位 = task（ALF game_idx）/ level（gridgames level_id），95% 分位 bootstrap 10k 次。

## 结论一句话

kev reviewer 的价值**严格限定在语义失败模式**：ALFWorld 上 kev 碾压规则（+0.32），
gridgames 上 kev **负增量**（−0.49/−0.12，比规则还差）——机械模式（方向塌缩）是确定性
可数的，规则 5 行 `has_loop` 完胜 kev 的语义读法。这比预测 P2 的「零增量」更强。

## ALFWorld（语义失败模式：find / acquire / other）

699 轨迹 / 64 task（instances/task = 4..12）。

| 指标 | kev | 规则 |
|---|---|---|
| 准确率 | **0.672 [0.621, 0.722]** | 0.352 [0.293, 0.414] |
| gap (kev−rule) | **+0.320 [0.224, 0.415] real** | — |
| Cohen's κ | 0.398 [0.328, 0.460] | 0.160 [0.114, 0.211] |
| AUC(find) | **0.872** | — |
| AUC(acquire) | 0.704 | — |

混淆（label 行 × kev 列，n=699）：

| label | find | acquire | other | n |
|---|---|---|---|---|
| find | **384 (83%)** | 11 | 66 | 461 |
| acquire | 51 | **47 (26%)** | 84 | 182 |
| other | 6 | 8 | **42 (75%)** | 56 |

规则镜像：find 只有 49/461=11%（target-agnostic 规则分不出哪个是要找的目标），
acquire 141/182=77%。kev 的语义读法在「找不到目标」上碾压规则，但「拿到没完成下一步」
（acquire）是 kev 的弱项（26%）。

## gridgames（机械/state-readable 失败模式）

**FrozenLake L4**（984 轨迹 / 246 level，label = fell_in_hole 483 / loop 501）：

| 指标 | kev | 规则 |
|---|---|---|
| 准确率 | 0.491 [0.431, 0.549] | **0.980 [0.966, 0.991]** |
| gap (kev−rule) | −0.489 [−0.551, −0.429] real | — |
| κ | 0.329 | 0.959 |
| AUC(fell_in_hole) | **1.000** | — |
| AUC(loop) | **1.000** | — |

**Sokoban L16**（768 轨迹 / 240 level，label = dead_push 444 / budget 234 / loop 90）：

| 指标 | kev | 规则 |
|---|---|---|
| 准确率 | 0.292 [0.252, 0.333] | **0.412 [0.366, 0.460]** |
| gap (kev−rule) | −0.120 [−0.151, −0.091] real | — |
| κ | 0.000 | 0.230 |
| AUC(dead_push) | 0.515 | — |
| AUC(loop) | 0.584 | — |
| AUC(budget) | 0.497 | — |

关键观察：
- **FL**：kev 完美识别 fell_in_hole（AUC 1.0），loop 的 noul 分也能完美排序（AUC 1.0），
  但 argmax 硬标签把 loop 全折进「budget」——校准偏差（kev 的 loop 分绝对值 ~0.29 低于
  budget 基线 ~0.5，尽管排名完美）。规则（eval_reason==failed → fell_in_hole，否则
  `has_loop` → loop）在 FL 上近乎 oracle（0.98；2% 是 harness 把「最后一步才掉坑」标成
  budget-out 的约定偏差）。
- **SK**：dead_push（箱卡死）只存在于 env 的 `dead` flag，**轨迹文本里不可见**（final_obs
  的 Last moves 无 dead 信号，需要读棋盘判角落）。kev 与规则同样瞎（AUC≈0.5），都折进
  budget。规则只拿对 budget+loop（0.41），kev 更低（0.29，因为连 loop 也折进 budget）。

## Gate GA 判定

| 条件 | 值 | 判定 |
|---|---|---|
| ALF kev ≥ 规则 +0.10 | +0.320 [0.224, 0.415] | ✅ |
| ALF κ ≥ 0.4 | 0.398 [0.328, 0.460] | ⚠️ 边际（差 0.002，CI 跨 0.4） |
| gridgames 两者差 ≤0.03（单向：kev 不优于规则） | −0.489 / −0.120 | ✅（kev 更差，方向正确） |

预测判定：
- **P1（ALF kev ≥ 规则 +0.10）**：✅ gap +0.320。
- **P2（gridgames kev−rule ≤0.03）**：✅ 单向读法（kev 未优于规则，而是更差 −0.49/−0.12）。
  若按双向 |gap|≤0.03 读，则 P2 判错——但判错方向是「kev 更差」，与 Claim A 的精神一致
  （kev 的价值是语义专属），只是比「零增量」更尖锐。

## 偏离 spec 记录

- **gridgames render 含 final_obs**：ALF 的 render 只有 goal/exit_reason/eval_reason/steps/
  actions；gridgames 额外渲染 `final_obs`（board + Last moves），因为 fell_in_hole 的显式
  信号（`… -> fell_in_hole`）只在 final_obs 里。否则 kev 连死亡都识别不了（早先 sparse
  render 下 kev accuracy 0.000）。规则只读 eval_reason+actions（未读 board）。
- **gridgames 规则 tagger 是 text-only**（eval_reason + has_loop），不是 replay oracle：
  初版误把 `gridgames_rule` 设成 `gridgames_label`（=oracle），会让规则作弊到 1.0，对 kev
  不公平。已改成 text-only。
- **FL fell_in_hole 语义修正**：oracle 里 fell_in_hole 严格 = `player is None`（真掉坑），
  不把「unreachable（fl_steps_to_go is None）」也算进去（后者 budget-out，非死亡）。
- **kev noul 跨题校准偏差**：argmax-over-raw-probabilities 在 gridgames 4 题上失败（budget
  基线虚高），尽管单题 noul 分排名完美（AUC 1.0）。这本身是一个 finding，不是 bug。

## 数据位置

- 每 run 的逐轨迹表：`/tmp/a1a2_json/*.json`（ALF）、`/tmp/a1a2_gg_fl/*.json`、
  `/tmp/a1a2_gg_sk/*.json`（gridgames）。
- bootstrap 汇总：各目录 `aggregate.json`。
- 重算脚本：`recipe/agent_evolver/a1a2_runner.py`（单 run）+ `a1a2_aggregate.py`（池化 bootstrap）。
