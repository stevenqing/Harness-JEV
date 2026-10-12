# ScienceWorld baseline — S0–S2 接入 + G0/G1/G2/G3/G4

环境 `scienceworld==1.2.3`（py3.8 `ee-sciworld`，Java 在 `ee-sciworld/bin/java`）。只做接入和 baseline，不进化、不训练。SPEC 在 `recipe/agent_evolver/SCIWORLD_SPEC.md`。

## 状态一览

| 门 | 状态 | 结果 |
|---|---|---|
| S0 任务选择 | ✅ | 12 类任务选中（见 G0） |
| S0 切分 | ✅ | heldin 64 + heldout 64（无重叠，128 局） |
| S0 环境接线（env server + client wrapper） | ✅ | 端到端冒烟通（create/step/invalid/goal_progress/destroy） |
| G0 可复现 | ⚠️ 改判 | 见下「G0：gold planner 非确定性」 |
| G1 环境正确 | ✅ | gold 重放 128/128 = 1.000（≥0.95）；random floor mean 0.0667 / success 0% |
| G2 标签正确 | ✅ | 抽 20 局失败轨迹人工核对 17/20 强一致（0.85）+3 边缘，≥0.85 过 |
| G3 已发表对照 | ✅ | Qwen3.5-4B held-in mean 0.4242 ∈[0.05,0.70] 过 |
| G4 baseline（4B/2B/0.8B） | ✅ 4B·2B 完成 / 0.8B 停 | 4B held-in 0.4242 ∈[0.10,0.60] 过；见下 |

## G0：gold planner 非确定性（改判说明）

`select_tasks.py` 复跑无法得到相同 `task_selection.json`，根因是 **1.2.3 的 gold path 生成非确定**：planner 按身份 hash 顺序搜索，同一 (task, variation) 在同一 JVM 内 reload 也给不同长度（实测 `chemistry-mix` var0 → `[23, 23, 20]`），跨 JVM 更不稳（`paint-tertiary` var0 → 17 vs 30）。SPEC §1 已点名此因：`1.3.0（未发布）才会把对象顺序改成确定性的 UUID 顺序`。

**但真正的环境 rollout 是确定的**：`load+reset` 的 `observation` / `valid` / `taskDesc` 同 JVM、跨 JVM 完全一致。非确定性只出在 gold planner，影响的是 gold 长度中位数（任务选择）和 §8.1 gold coverage 度量，不影响 G1 的 gold 重放（无论哪条 gold path 都稳定重放到 100）。

**决定（用户拍板「锁死工件 + 含 borderline tertiary」）**：把 `task_selection.json` + `splits.json` 当一次性生成、提交锁死的工件，下游 G1–G4、三臂全用这个固定 split。唯一骑在阈值（gold 中位数 24~28 抖，重放稳定 100，~25 步在 40 步预算内）的 `chemistry-mix-paint-tertiary-color` 保留。

**锁定工件 sha256：**

```
task_selection.json  147381cda29bd38e869c2918b7a67db8349ac6f2048e29c7281e3f8a0c0450b8
splits.json          4992ee6ef00d5eb8a77a7d08d4fb9fbc14a39a2059f243c9c265a64f0da82842
```

### 选中的 12 类任务

| 任务 | gold 长度中位数 | 重放分数 |
|---|---|---|
| chemistry-mix | 16 | [100×5] |
| chemistry-mix-paint-secondary-color | 21 | [100×5] |
| chemistry-mix-paint-tertiary-color | 22 | [100×5] |
| find-animal | 10 | [100×5] |
| find-living-thing | 10 | [100×5] |
| find-non-living-thing | 5 | [100×5] |
| find-plant | 10 | [100×5] |
| identify-life-stages-2 | 14 | [100×4]（只有 4 个 train 变体） |
| lifespan-longest-lived | 6 | [100×5] |
| lifespan-longest-lived-then-shortest-lived | 7 | [100×5] |
| lifespan-shortest-lived | 6 | [100×5] |
| use-thermometer | 20 | [100×5] |

排除：electrical 4 类（`noElectricalAction` 下 `load` 抛 `ValueError`，SPEC §3.1 预期）；其余 13 类 gold 中位数 >25（boil 41 / melt 42 / freeze 37 / grow-plant 67 / grow-fruit 81 / mendelian 129 / inclined-plane-* 53–180 等）；`identify-life-stages-1` 中位数 30。

### 切分分配（heldin / heldout 各 64）

`identify-life-stages-2` 只有 4 train + 4 test 变体，不足 base 份额；`splits._distribute` 做了 cap + 再分配（短额给变体最多的 `use-thermometer`），保住了 64+64。

| 任务 | heldin | heldout |
|---|---|---|
| chemistry-mix | 5 | 5 |
| chemistry-mix-paint-secondary-color | 5 | 5 |
| chemistry-mix-paint-tertiary-color | 5 | 5 |
| find-animal | 6 | 6 |
| find-living-thing | 6 | 6 |
| find-non-living-thing | 6 | 6 |
| find-plant | 5 | 5 |
| identify-life-stages-2 | 4 | 4 |
| lifespan-longest-lived | 5 | 5 |
| lifespan-longest-lived-then-shortest-lived | 5 | 5 |
| lifespan-shortest-lived | 5 | 5 |
| use-thermometer | 7 | 7 |

## G1：环境正确

- **gold 重放**：128/128 局重放对应变体的 gold 序列后 `final_score == 100`，**1.000**（门 ≥0.95，PASS）。无失败局，无需逐条列因。
- **random floor**：从当前合法动作组合均匀抽样，每局 2 次、最多 40 步。mean score **0.0667**（0–1），success rate **0.0**。符合 SPEC §7「小模型成功率贴地板」的预期，floor 极低。

## 环境接线（S0 冒烟）

- `benchmarks/sciworld/env_server.py`（py3.8，16-worker spawn 池，默认 :18091）+ `envs/sciworld/env.py`（`.venv` 薄 HTTP client）端到端冒烟通过：
  - `reset` → task_name/variation/obs（任务描述 + look）/possible_actions(24)/possible_objects(16)
  - `step("look around")` → in_valid=True / score 8 / moves 0
  - `step("zzz not a real action")` → in_valid=False / obs=`No known action matches that input.`（`invalid_action` 标签依赖的精确文案）
  - `goal_progress` / `destroy` 正常；跑完无残留 `java` 进程。

### `get_goal_progress()` 解析前后（SPEC §8.1 要求）

原始（节选）：

```
Completed keys:
----------------------------------------------------------------------------------------------------
Sequential Subgoals:
----------------------------------------------------------------------------------------------------
0	false	                       GoalFocusOnAnimal	focus on an animal
1	false	                   GoalObjectInContainer	move living thing to answer box
----------------------------------------------------------------------------------------------------
Unordered and Optional Subgoals:
----------------------------------------------------------------------------------------------------
0	false	                   GoalObjectInContainer	Pick up object
1	true	                  GoalInRoomWithOpenDoor	Be in a room with an open door
...
```

解析后：

```
ordered   = [{index:0, completed:False, type:GoalFocusOnAnimal,  name:"focus on an animal"},
             {index:1, completed:False, type:GoalObjectInContainer, name:"move living thing to answer box"}]
unordered = [{index:0, completed:False, type:GoalObjectInContainer, name:"Pick up object"},
             {index:1, completed:True,  type:GoalInRoomWithOpenDoor, name:"Be in a room with an open door"}, ...]
```

`first_unmet_subgoal` 取 ordered 里第一个未完成的 name（本例 "focus on an animal"）。

## 代码与产物

- `envs/sciworld/{select_tasks,splits,env,projection,failure_labels,render,gold_agent,random_agent}.py` + `vendor/`（RLVMR 参考，见 `vendor/VENDOR.md`）
- `benchmarks/sciworld/env_server.py`
- `agents.py` +`SCIWORLD_TEMPLATE(_NO_HIS)` / `run_sciworld`；`run.py`/`trajectories.py`/`spec.py`/`native/config.py` 注册 `sciworld`
- `three_axis_sciworld.sh`（三臂 shell，`JUDGE=meta` 默认，:18091）
- 产物：`task_selection.json`、`splits.json`（sha256 见上）

## G2：标签正确（人工核对 20 局）

抽 20 局失败 baseline 轨迹（4B held-in，按 episode_label 分层），逐条人工核对整局标签：

| auto 标签 | 样本 | 强一致 | 边缘 |
|---|---|---|---|
| wrong_focus | 6 | 5 | 1 |
| stalled_partial | 6 | 6 | 0 |
| loop | 3 | 2 | 1 |
| invalid_dominant | 3 | 3 | 0 |
| timeout_partial | 2 | 1 | 1 |
| **合计** | **20** | **17** | **3** |

**一致率 = 17/20 = 0.85**（≥0.85，PASS）。3 个边缘非错标，而是 `failure_labels.py` 的 `wrong_focus` 优先级压在 loop/stalled 之前的排序取舍（例：`chemistry-mix-paint-tertiary-color_0` 前 29 步纯绕圈、末步才 focus 错，被归 wrong_focus 而非 stalled；`use-thermometer_1` 更像 invalid 主导但末 6 步 distinct≤2 判 loop）。记为已知 nuance，不修规则。

## G3：与已发表数字对照

Qwen3.5-4B held-in（64 局，R0 baseline）：mean score **0.4242**、success **14/64 (0.219)**。门 ∈[0.05,0.70]，**PASS**，不必贴轨迹查因。published 参考（Qwen2.5-7B easy、30 步）score 17–28，模型/步数/任务子集都不同，不对齐。

## G4：baseline（4B/2B/0.8B × heldin/heldout）

### 总表（mean score / success）

| 模型 | held-in | held-out |
|---|---|---|
| 4B | 0.4242 (14/64) | 0.3281 (10/64) |
| 2B | 0.2044 (5/64) | 0.1388 (2/64) |
| 0.8B | 停（非关键路径） | — |

### 4B（完成）

| split | mean score | success | pass_rate |
|---|---|---|---|
| held-in | 0.4242 | 14/64 | 0.219 |
| held-out | 0.3281 | 10/64 | 0.156 |

门：4B held-in ∈[0.10,0.60] → **0.4242 PASS**。held-out 0.3281 保留 77%（无过拟合，与 ALF/WS 基线一致）。

**episode_label 分布**：

| label | held-in | held-out |
|---|---|---|
| success | 14 | 10 |
| wrong_focus | 20 | 24 |
| stalled_partial | 22 | 17 |
| loop | 3 | 4 |
| invalid_dominant | 3 | 4 |
| timeout_partial | 2 | 4 |
| stalled_zero | 0 | 1 |

**first_unmet_subgoal top（两 split 合并）**：move living thing to answer box（23）、focus on instrument（13）、focus on (intermediate) mixing result / violet·green paint（约 15）、focus on an animal / a living thing（约 12）、focus on animal with shortest life span（10）。注：约 20–30 条轨迹 ordered subgoal 已全完成但仍失败（first_unmet_subgoal 为空），说明瓶颈不只在第一个未完成子目标。

**按任务类型 mean score**（最弱→最强）：

| 任务 | held-in | held-out |
|---|---|---|
| chemistry-mix-paint-tertiary-color | 0.006 | 0.000 |
| use-thermometer | 0.034 | 0.026 |
| chemistry-mix | 0.158 | 0.214 |
| identify-life-stages-2 | 0.163 | 0.100 |
| lifespan-longest-lived-then-shortest-lived | 0.200 | 0.200 |
| find-living-thing | 0.430 | 0.375 |
| chemistry-mix-paint-secondary-color | 0.460 | 0.040 |
| find-animal | 0.500 | 0.292 |
| lifespan-shortest-lived | 0.700 | 0.600 |
| lifespan-longest-lived | 0.800 | 1.000 |
| find-plant | 0.814 | 0.316 |
| find-non-living-thing | 0.832 | 0.762 |

### 2B（完成）

| split | mean score | success | pass_rate |
|---|---|---|---|
| held-in | 0.2044 | 5/64 | 0.078 |
| held-out | 0.1388 | 2/64 | 0.031 |

**episode_label 分布**：held-in = wrong_focus 23 / invalid_dominant 14 / loop 14 / stalled_partial 6 / success 5 / timeout_partial 1 / stalled_zero 1；held-out = wrong_focus 25 / loop 15 / invalid_dominant 13 / stalled_partial 4 / stalled_zero 3 / timeout_partial 2 / success 2。与 4B 相比 `invalid_dominant`/`loop` 大幅上升（2B 更频繁产出非法动作和循环），`stalled_partial` 下降（2B 更难做出有效进度）。

**按任务类型 mean score（最弱→最强）**：

| 任务 | held-in | held-out |
|---|---|---|
| chemistry-mix-paint-tertiary-color | 0.000 | 0.000 |
| use-thermometer | 0.009 | 0.004 |
| lifespan-longest-lived-then-shortest-lived | 0.034 | 0.034 |
| identify-life-stages-2 | 0.080 | 0.163 |
| chemistry-mix-paint-secondary-color | 0.120 | 0.100 |
| find-animal | 0.137 | 0.027 |
| find-plant | 0.150 | 0.064 |
| lifespan-longest-lived | 0.150 | 0.350 |
| find-living-thing | 0.167 | 0.013 |
| chemistry-mix | 0.224 | 0.098 |
| find-non-living-thing | 0.582 | 0.247 |
| lifespan-shortest-lived | 0.800 | 0.650 |

### 0.8B（停）

用户拍板（2026-10-10）：0.8B 非三臂关键路径且极慢（held-in >78min，thinking-on + 高频 invalid 循环，比 2B 慢 2 倍+，属模型行为非卡死），直接停掉、立刻上 4B 三臂。规模扫描完整性交予末尾，不影响三臂结论。

## 待办

- 每局 JSON 汇总到 `runs/sciworld_baseline/`（SPEC §10 交付物）。
- **ScienceWorld 三臂（llm/prior/enforce，4B，JUDGE=meta）已启动**（2026-10-10 15:10，`three_axis_sciworld.sh`，sw3x 前缀，6 轮 held-in + held-out probe）。跑完补三臂结果节。
- TextCraft 三臂（`three_axis_textcraft.sh` 需 `JUDGE=meta` 覆盖默认 kev）押后。
