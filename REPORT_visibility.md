# Visibility story — 报告（进行中）

> 生成于 2026-10-08。全程不训练/不改任何模型权重；kev 原样用，semif 是冻结 readout。
> 本报告覆盖已完成的 S2 基座 + Track A(A1/A2 gate GA + kev 先验注入六轮) + Track B(B1)。
> **A3（三臂 3 seeds）已停 + rule 臂归档退役**（Phase 2a，见 §5），P3/P4 判 N/A。**B2 押后**（spec 缺失，见 §9）。
> **规模扫描（2B/0.8B 三轴）已停**（代码精简前主动 kill，0.8B WS 中止）：2B ALF/WS 已出、0.8B ALF 地板 0（三臂 6 轮全 0.000 + 4 探针全 0）——见 §4.5。

---

## 1. 基座（S1–S4）

### 1.1 解码设置（S1）

所有生成式 run 统一解码，实际值为：

| 项 | 值 |
|---|---|
| agent 模型 | Qwen3.5-4B（vLLM 4 卡 :8200–8203，冻结，永不训练） |
| thinking | **ON**（`<think>…</think>` 前缀，extract_action 只读 `<action>`） |
| max_tokens | **4096** |
| agent temperature | **0.0**（`EVOLVER_AGENT_TEMPERATURE=0`，确定性 rollout；覆盖 `spec.py` 默认 0.4） |
| agent top_p | 0.8（默认） |
| ALF/WS max_steps | 50 / 15 |
| history_length | 2 |
| meta 模型 | deepseek-v4-pro（mioffice anthropic 通道，extended_thinking，temperature=1） |

> **偏离记录**：`spec.py` 的 `AGENT_TEMPERATURE` 默认是 0.4（Qwen 推荐值），但本 spec 的 run（A3 三臂、早先 GA 的三轴对照）统一用 `EVOLVER_AGENT_TEMPERATURE=0` 做确定性 rollout——否则跨 run 的 vLLM bf16/重启噪声会污染逐 seed 的 arm 对比（见 memory `harnessx-jev-opened-o-negative`）。早期 thinking-off run 一律不复用。

### 1.2 每 env 一个完整 prompt（S1/§6）

**ALFWorld**（`agents.py:ALFWORLD_TEMPLATE`，system = Qwen3.5 默认 system，guidance 槽 base 为空）：

```
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
{skills}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
```

**WebShop**（`agents.py:WEBSHOP_TEMPLATE`）：同构，`admissible_actions` 换成换行列表，task/obs 措辞换 WebShop。

**gridgames readout（B1）**（`r0.py:system_message` + `b1_readout._QUESTION`）：system = 游戏规则+符号表+步数预算；user = 当前棋盘 + 剩余步数 + 最近 3 步及结果；readout 题固定：

> Choose the single move (up, down, left, or right) that best advances toward the goal.
> options: up / down / left / right

### 1.3 bundle S2（G0/G1）

- `verify_levels.py`（G0）：**G0 OK 3584 levels（139776 random moves compared）**。
- `reference_run.py`（G1a–d）：stub 模型走全 pipeline，**全绿**：
  - **G1a NullStub**（uniform 随机，16/level）PASS，calls=151916，parse_fail=0（success 贴近 floors：FL L4 0.0479 / SK L4 0.0654，L8+ 近 0）；
  - **G1b SilentStub** PASS（calls=15360，bad=[]）；**G1c SolverStub** PASS（calls=5352，bad=[]）；**G1d ScriptStub** 产出 512 reports（calls=10075）。
- 3584-level manifest、per-cell 切分 evolve 256 / gate 64 / test 128、episode contract（`episode_seed=1000*i+k`、`request_seed=100*episode_seed+m`）。

---

## 2. Track A — A1/A2（gate GA）

> 完整逐轨迹表与重算见 `recipe/agent_evolver/A1A2_RESULTS.md`。采样单位 = task（ALF game_idx）/ level（gridgames level_id），95% 分位 bootstrap 10k 次。

**结论**：kev reviewer 的价值**严格限定在语义失败模式**。ALFWorld 上 kev 碾压规则（+0.32）；gridgames 上 kev **负增量**（比规则还差）。

### 2.1 ALFWorld（find / acquire / other，699 轨迹 / 64 task）

| 指标 | kev | 规则 |
|---|---|---|
| 准确率 | **0.672 [0.621, 0.722]** | 0.352 [0.293, 0.414] |
| gap (kev−rule) | **+0.320 [0.224, 0.415] real** | — |
| Cohen's κ | 0.398 [0.328, 0.460] | 0.160 [0.114, 0.211] |
| AUC(find) / AUC(acquire) | **0.872** / 0.704 | — |

混淆（label 行 × kev 列，n=699）：find 384/461=83%、acquire 47/182=26%、other 42/56=75%。规则镜像 find 49/461=11%（target-agnostic 分不出要找的目标）。

### 2.2 gridgames（state-readable 失败模式）

| cell | kev acc | 规则 acc | gap (kev−rule) | 说明 |
|---|---|---|---|---|
| FL L4（984 轨迹，fell_in_hole 483 / loop 501） | 0.491 | **0.980** | −0.489 real | kev AUC(fell_in_hole)=1.0、AUC(loop)=1.0，但 argmax 硬标签因校准偏差把 loop 折进 budget |
| SK L16（768 轨迹，dead_push 444 / budget 234 / loop 90） | 0.292 | **0.412** | −0.120 real | dead_push 只在 env `dead` flag、轨迹文本不可见，kev/规则都瞎 |

### 2.3 Gate GA 判定

| 条件 | 值 | 判定 |
|---|---|---|
| ALF kev ≥ 规则 +0.10 | +0.320 [0.224, 0.415] | ✅ |
| ALF κ ≥ 0.4 | 0.398 [0.328, 0.460] | ⚠️ 边际（差 0.002，CI 跨 0.4） |
| gridgames 差 ≤0.03（单向：kev 不优于规则） | −0.489 / −0.120 | ✅（kev 更差，方向正确） |

---

## 3. Track B — B1（readout R0，无进化）

> readout agent = 每步给当前棋盘 + 四动作选项做一次前向打分、取 argmax，无自回归。8 格 gate split、每 level 4 episodes、每 backend 2048 episodes / 0 left_out。

### 3.1 结果：`success − floors.gate.uniform` per cell

| game | tier | floor | kev succ (Δ) | **semif succ (Δ)** | semif 95% CI |
|---|---|---|---|---|---|
| frozenlake | L4 | 0.0569 | 0.094 (+0.037) | **0.125 (+0.068)** | [0.047, 0.203] |
| frozenlake | L8 | 0.0077 | 0.000 (−0.008) | **0.016 (+0.008)** | [0.000, 0.047] |
| frozenlake | L16 | 0.0020 | 0.000 (−0.002) | **0.016 (+0.014)** | [0.000, 0.047] |
| frozenlake | L32 | 0.0000 | 0.000 (+0.000) | 0.000 (+0.000) | [0.000, 0.000] |
| sokoban | L4 | 0.0688 | 0.016 (−0.053) | **0.094 (+0.025)** | [0.031, 0.172] |
| sokoban | L8 | 0.0125 | 0.000 (−0.013) | 0.000 (−0.013) | [0.000, 0.000] |
| sokoban | L16 | 0.0020 | 0.000 (−0.002) | 0.000 (−0.002) | [0.000, 0.000] |
| sokoban | L32 | 0.0000 | 0.000 (+0.000) | 0.000 (+0.000) | [0.000, 0.000] |

### 3.2 关键发现

- **semif（冻结 base 的 first-token logit readout）全面反超 kev（训练过的 LoRA+pointer head）**。kev 在 SK L4 甚至掉到 floor 以下（−0.053）。这与 memory `semif-direct-logit-eval`、`system1-scoring-eval` 的排序互相印证：训练头学到的信号对**逐步闭集四选一**无帮助，原始 logit 反而更稳。
- readout 曲线的真实起点：只有 FL L4/L8/L16 与 SK L4 出 floor；**SK L8 停在 floor（−0.013）**，是 B2 要攻的目标（P6 要 ≥ +0.10 才有戏）。

### 3.3 P5 判定

- **P5（Sokoban L8 readout R0 在 uniform floor ±0.05 内）**：✅ kev −0.013、semif −0.013，都在 ±0.05 内。

---

## 4. Track A — kev 先验注入（六轮全量：llm / prior / enforce + 基线）

> kev 当 reviewer 注入 = 给失败轨迹打「失败机制」分，作为先验塞进 deepseek meta 的 brief。三组已完成的 temp=0 / 64-task 对照，全部 6 轮 + held-out 探针。词表两代：**Mode A**（benchmark 专属 find/acquire）与**三轴**（通用 lens/lever/intent）。

> **词表口径（重要）**：本报告所有「kev 当 decision prior 注入」的跑——§4.1 ALF 三轴、§4.2 WS 三轴、§5 A3 的 kev 臂，以及进行中的 2B/0.8B 三轴规模扫描——kev 打分用的词表是 **general 三轴**（`lens_failure` / `lens_capability_gap` / `lever`，定义在 `decision_prior.py::AXIS_QUESTIONS`），**不是 benchmark 专属词表**。benchmark 专属的 `find/acquire/other` 只出现在三处、且都不是 kev 的注入词表：
> 1. **§2 A1/A2 gate GA**——`find/acquire/other` 是从 ALFWorld expert plan（`traj_data.json` 的 `plan.high_pddl`）反推的 **ground-truth 标签**，用来给 kev reviewer 打准确率/κ/AUC（考官，不是选手）。
> 2. **§5 A3 的 `rule` 臂**——A2 规则 tagger 产出的 `find/acquire/other` 是**规则 baseline 对照**，与 kev 无关。
> 3. **§4.3 Mode A**——唯一一次用 benchmark 专属词表（`find/acquire`）做 kev 注入的历史对照，已被三轴取代。

### 4.1 ALFWorld — 三轴（lens/lever/intent 通用词表）

**held-in 六轮弧线（pass rate）**：

| round | llm | prior | enforce |
|---|---|---|---|
| R0 | .344 | .344 | .344 |
| R1 | .406 | .312 | .406 |
| R2 | .469 | .516 | .328 |
| R3 | .578 | .562 | .531 |
| R4 | .578 | .656 | .359 |
| R5 | **.734** | **.672** | **.531** |

**held-out 探针（各臂 held-in best config 重放，64 task）**：

| arm | held-in best | held-out | 泛化 Δ |
|---|---|---|---|
| baseline | .344（R0） | .297 | −.047 |
| llm | .734（R5） | .781 | +.047 |
| **prior** | .672（R5） | **.844** | **+.172** |
| enforce | .531（R3） | .594 | +.063 |

### 4.2 WebShop — 三轴（官方 train=held-in / test=held-out）

**held-in 六轮（pass / mean_reward）**：

| round | llm | prior | enforce |
|---|---|---|---|
| R0 | .156 / .425 | .156 / .425 | .156 / .425 |
| R1 | .125 / .335 | .297 / .577 | .266 / .562 |
| R2 | .188 / .502 | .266 / .565 | .219 / .486 |
| R3 | .281 / .536 | .203 / .581 | .234 / .530 |
| R4 | .266 / .515 | **.312 / .598** | .266 / .562 |
| R5 | .281 / .526 | .250 / .462 | **.250 / .562** |

**held-out（pass / mean_reward）**：

| arm | held-in best | held-out |
|---|---|---|
| baseline | — | .203 / .439 |
| llm | .281 / .536（R3） | .312 / .560 |
| **prior** | .312 / .598（R4） | **.344 / .630** |
| enforce | .250 / .562（R5） | .203 / .567 |

### 4.3 ALFWorld — Mode A（find/acquire 词表，最早 temp=0 干净 A/B）

**held-in 六轮弧线（pass rate）**：

| round | llm | kev |
|---|---|---|
| R0 | .344 | .344 |
| R1 | .328 | .422 |
| R2 | .516 | .656 |
| R3 | .734 | .812 |
| R4 | .719 | .844 |
| R5 | .766 | **.906** |

**held-out**：baseline .297 → llm best .797 → **kev best .969**（+17pt over llm）。

### 4.4 小结

- **prior 三组全 first**：ALF 三轴 held-out .844、WebShop 三轴 .630、Mode A（kev） .969。
- **零过拟合主要在 prior/llm**：prior ALF held-out .844 > held-in .672、Mode A .969 > .906、WS .344 > .312；llm 亦 held-out ≥ held-in。enforce 不赢也不明显过拟合（ALF .594、WS .567 ≈ held-in best）。
- **enforce（硬绑定）不稳健**：ALF .594 输 llm .781、WS .567 ≈ llm .560。软先验可被覆盖，硬裁决砍掉强模型搜索空间——「硬绑定优于软先验」假设不被支持。
- 数据位置/重算见 §8，机制与定位见 §10。

### 4.5 规模扫描（2B / 0.8B 三轴）——已停

> 同一三轴 kev 先验注入（llm / prior / enforce），只把被进化的 agent 底座从 Qwen3.5-4B 换成 **2B / 0.8B**（kev reviewer :8090 不变）。问的是「kev 先验增益是否随 agent 规模缩放」。脚本 `archive/scripts/three_axis_scale_2b_0.8b.sh`（已归档），64 worker（K=16/卡）batching，temp=0 确定性 rollout；held-in 6 轮 + 4 个 held-out 探针（baseline / llmbest / priorbest / enforcebest，best = held-in `mean_reward` 最大轮，`--base-config` 重放）。held-in/held-out 口径同 §4.1（ALF in/out-of-distribution）与 §4.2（WS 官方 train=held-in / test=held-out）。
>
> **口径**：held-out 探针的 comparison.json `config` 字段恒为 `baseline`（run.py 里 round_idx=0 的标签占位），但实际重放的是 `--base-config` 传入的 held-in best 配置——已核对非 bug。batching 下 temp=0 仍会翻转 ~2/64 近 tie task（±3% 绝对噪声），对下面这些贴地板的小 n 数字要打折读。

#### 4.5.1 2B ALF（二值 reward = pass rate，64 task）

| arm | held-in best（轮） | held-out（pass） |
|---|---|---|
| baseline | .031 (R0) | .047 (3/64) |
| llm | .172 (R3) | .141 (9/64) |
| **prior** | .156 (R1) | **.172 (11/64)** |
| enforce | .125 (R5) | .125 (8/64) |

全在地板附近（pass 个位数），排序 prior > llm > enforce 勉强成立，但离 4B 锚点（§4.1 held-out .844/.781/.594）差一个数量级。

#### 4.5.2 2B WS（连续 reward，64 task）

| arm | held-in best（轮） | held-out reward（pass） |
|---|---|---|
| baseline | .473 (R0) | .483 (.141) |
| llm | .473 (R0) | .504 (.172) |
| **prior** | .477 (R0) | **.537 (.188)** |
| enforce | .492 (R5) | .440 (.125) |

排序 prior > llm > baseline > enforce，与 4B 锚点（§4.2 held-out .630/.560）一致。注意：llm / prior 的 held-in best 都是 **R0 baseline**（进化在 mean_reward 上回退，无 held-in 增益可泛化），只有 enforce 真进化出 R5 增量（.431→.492）——可它的 held-out 仍最差。

#### 4.5.3 0.8B（ALF 地板 / WS 中止）

- **ALF（已跑完）**：llm / prior / enforce 三臂 **6 轮 held-in 全 0.000**（mean_reward 0），4 个 held-out 探针（baseline/llmbest/priorbest/enforcebest）pass_rate **全 0.0** —— 0.8B ALF 撞二值地板（一个 task 都过不了，无失败模式可供 kev 标注），三轴 kev 先验在零信号上无效。
- **WS（中止）**：主动 kill 扫描时 llm 臂还在 meta 中途，`ws3x_0_8b_llm_heldin/` 只留下半截轨迹、无 comparison.json，其余两臂未启动。**0.8B WS 无结果**。

#### 4.5.4 规模扫描小结（三判断）

1. **绝对能力随规模塌**：4B ALF prior .844 → 2B .17 → 0.8B 0；kev 先验救不了二值地板上的模型。
2. **kev 相对增益也缩**：4B 上 prior−llm ≈ +.06~.07，2B 上只剩 +.03（ALF +.031 / WS +.033），量级掉进 64-task 噪声。
3. **enforce 持续最弱**：所有有信号的 cell 里 enforce held-out 全垫底/近垫底，与 4B 结论「硬绑定不赢」一致。

---

## 5. Track A — A3（三臂 3 seeds）—— 已停 + rule 臂归档退役

状态：**已停 + rule 臂归档（Phase 2a）**。A3 启动于 2026-10-08 15:36（llm/kev/rule 三臂 × 3 seeds 1234/2345/3456），因规模扫描优先而停（`kill -TERM -26689`），最终未续跑。Phase 2 代码精简把 rule 注入臂整线归档：`rule_prior.py` + `a3_three_arm_3seed.sh` → `archive/rule_arm/`，并从 `run.py` 删掉 `--decision-backend rule` 分支。

- `llm`（无 decision backend）/ `kev`（`--decision-backend kev --decision-mode prior`）三轴先验，单 seed 结果已由 §4.1（ALF）/ §4.2（WS）覆盖；A3 的多 seed 稳健性未跑。
- `rule` 臂（`--decision-backend rule`，A2 规则 tagger 注入同一 brief 通道）退役。kev-vs-rule 的 tag 级对比由 §2 A1/A2 gate GA 提供（ALF kev .672 vs rule .352）。

> 影响：§6 的 P3/P4（held-out kev ≥ rule +0.05）不再由 A3 出结果 → 判 N/A。

---

## 6. 预测判定（rollout 前冻结）

| # | 预测 | 状态 |
|---|---|---|
| P1 | ALFWorld kev lens 准确率 ≥ 规则 +0.10 | ✅ +0.320 |
| P2 | gridgames kev−rule ≤0.03 | ✅ 单向（kev 更差 −0.49/−0.12；比「零增量」更尖锐） |
| P3 | ALFWorld held-out kev ≥ rule +0.05 success | — A3 rule 臂退役（Phase 2a），N/A |
| P4 | WebShop held-out kev ≥ rule +0.03 reward | — A3 rule 臂退役（Phase 2a），N/A |
| P5 | Sokoban L8 readout R0 在 floor ±0.05 内 | ✅ −0.013 |
| P6 | Sokoban L8 evolved readout 出 floor ≥ +0.10 | ⏳ B2 押后 |
| P7 | Sokoban L8 evolved gen 不出 floor +0.10 | ⏳ B2 押后 |
| P8 | FL-L16 evolved readout 在 evolved gen ±0.05 内 | ⏳ B2 押后 |
| P9 | C1 三臂等 rollout 下两两 ≤0.05 | ⏳ 未跑（C1 在 GB 后） |

---

## 7. What the data do not support

- **kev 不是更好的 readout**：B1 里 kev 在逐步四选一上被冻结 base 的 semif 全面反超，甚至在 SK L4 掉到 floor 以下。kev 的增量只在「轨迹语义失败模式」（A1/A2 的 ALF find），不延伸到逐步决策。
- **kev 的 κ 是边际的**（0.398，CI 跨 0.4）：GA 靠「+0.32 准确率」这一硬条件通过，κ 未干净过 0.4，不能拿「kev 与 oracle 高度一致」来叙事。
- **gridgames 上 kev 是负增量**（−0.49/−0.12）：不是「零增量」，是更差——机械/state-readable 失败模式规则 5 行 `has_loop` 完胜语义读法。这与 Claim A 精神一致，但方向比预测更尖锐。
- **SK-L8 readout 停在 floor**：B1 未证明 readout 能在 Sokoban 出 floor；Claim B 的「readout 能到 gen 到不了的 harness」尚未被支持，等 B2。

---

## 8. 数据位置 + 重算脚本

| 产物 | 位置 | 重算 |
|---|---|---|
| A1/A2 逐轨迹 | `/tmp/a1a2_json/*.json`、`/tmp/a1a2_gg_{fl,sk}/*.json` | `recipe/agent_evolver/a1a2_runner.py` + `a1a2_aggregate.py` |
| A1/A2 汇总 | `recipe/agent_evolver/A1A2_RESULTS.md` | — |
| B1 episodes | `runs/agent_evolver/gridgames/b1_readout/{kev,semif}/episodes.jsonl` | `recipe/agent_evolver/report_b1.py` |
| floors.gate.uniform | `gridgames/tiers.json` | — |
| A3 | `runs/evolve/a3_{llm,kev,rule}_s{seed}_{heldin,heldout}/comparison.json` | `recipe/agent_evolver/run.py` |
| 规模扫描 2B/0.8B | `runs/evolve/{ov3x,ws3x}_{2b,0_8b}_{llm,prior,enforce}_heldin/comparison.json` + `{baseline,llmbest,priorbest,enforcebest}_heldout/comparison.json` | `archive/scripts/three_axis_scale_2b_0.8b.sh` |

每 episode 一条 JSON（environment / backend / level-or-task / episode / `report()` 字段 / 每步概率向量+动作），所有数字可从 episodes + `levels.jsonl`/`tiers.json` 重算。

---

## 9. 偏离 spec 记录

- **agent temperature 0.0**（覆盖 `spec.py` 默认 0.4）：见 §1.1，为确定性 rollout。
- **B2 spec 缺失**：VISIBILITY_PLAN §3 引用的 `harnessx_jev_semif_spec.md` 磁盘上不存在（只找到 HarnessX-jev 的 `JEVTREE_SPEC.md`，是另一个 ALFWorld processor 合成设计 spec）。经用户拍板 **B2 押后**，先写本报告。
- **gridgames render 含 final_obs**、**规则 tagger 是 text-only 非 replay oracle**、**FL fell_in_hole 严格 = player is None**、**kev 跨题校准偏差是 finding 非 bug**：见 `A1A2_RESULTS.md` §偏离 spec 记录。

## 10. Discussion — kev 先验为什么有效 + 定位

### 10.1 机制（解释 prior > llm > enforce、held-out ≥ held-in）

核心反直觉点：kev 只有 0.67 的 tag 准确率（ALF A1/A2），deepseek 读轨迹明明更强，为什么注入 kev 的弱读反而更好？答案：meta 的瓶颈不是「能否理解」，而是「在什么认知成本 / 方差 / 偏置下理解」——kev 更便宜、更稳定、更抗偏置。

- **机制一（base-rate 去偏）**：deepseek 读 50+ 条 narrative 有系统性 base-rate 失真（戏剧性的 `find` 盖过安静的 `acquire`）；kev 是判别式逐条读 logit 再聚合，天然把 base rate 数对。kev 的 0.67 是均匀噪声，deepseek 未受助的误差是系统偏置——前者的分布更接近真相。
- **机制二（正则化 → 泛化）**：kev 的 tag 是语义抽象，锚定 meta 去写「失败机制」级 guidance（可泛化），而非「具体轨迹表面特征」（过拟合 held-in）。数据佐证：prior held-in best 更低（.672 < llm .734）但 held-out 反超（.844 > .781，泛化 Δ +.172 vs +.047）。
- **机制三（System-1/System-2 分工）**：deepseek 做开放合成（改什么），kev 做廉价一致判断（失败模式是什么）。enforce 跨了这条线——让弱判别模型做合成级绑定决策 + candidate reject，砍掉强模型的搜索空间还加 reject 代价 → enforce .594 < prior .844。
- **机制四（语义边界）**：上述机制成立条件是失败模式 (a) 有真信号 (b) 对 meta 昂贵 (c) 对单次前向读头廉价可近似。语义失败模式（ALF find/acquire、WS 属性点击）全满足；state-readable 机械模式（gridgames 掉坑/loop）全不满足 → kev 纯噪声（−0.49）。

**诚实标注**：以上是最自洽解释，尚未用独立消融钉死每个机制（如「同格式随机噪声分布 vs 真 kev 分布」来拆 de-variance vs 语义内容的贡献）。已由 rollout 证明的是 prior > llm > enforce、held-out ≥ held-in、A1/A2 准确率边界。

### 10.2 与 LLM-as-judge 的差别

| 维度 | LLM-as-judge | 我们的 kev |
|---|---|---|
| 判断强度 | 强判弱（GPT-4 判弱模型） | 弱判强（4B 读头判/喂 deepseek） |
| 角色 | verdict = ground-truth 代理 | fallible prior（可推翻） |
| ground truth | judge 本身 | 环境 rollout |
| 判的对象 | 产出质量（对错/好坏） | 失败机制（为什么错） |

关键证据：同一 kev，做成「软先验」赢（prior > llm）、做成「硬裁决」不赢（enforce ≤ llm：ALF .594<.781、WS .567≈.560）——即「judge as verdict」在本题被证伪。kev 0.67 还能有用，正因为它永远不越过「先验」这条线。

### 10.3 在 harness evolution 里的位置

标准 prompt/harness 进化循环 = 候选生成 → 评估(fitness) → 选择(gate) → 诊断(改什么) → 回生成。我们各阶段的实现 vs 场上默认：

| 阶段 | 我们 | 场上默认 |
|---|---|---|
| 候选生成 | deepseek meta | OPRO/EvoPrompt/ADAS 的 LLM 优化器（同） |
| 评估 fitness | 环境 rollout（唯一 ground truth） | task metric（DSPy metric / OPRO accuracy） |
| 选择 gate | score_and_gate（按 rollout） | 按 fitness 选 best |
| 诊断「改什么」 | **kev 先验（判别式读头）** | 跳过 / 生成式 self-critique（Promptbreeder/Reflexion）/ textual gradients（ProTeGi） |

**定位**：jev 是「诊断/critic」这一格把 System-2（强生成自省）换成 System-1（弱冻结判别读头）的实现；概念上对应 verifier-in-the-loop / weak-to-strong，但语义错位两点——verifier 判「对错」、jev 判「为什么错」；verifier 当 fitness 用、jev 从不当 fitness（fitness 永远是环境）。经验结论：harness evolution 的诊断模块，「判别式弱读头 + 软先验」比「生成式强自省 + 硬裁决」更稳。

## 11. 安全

token/key/secret 不进 report/log/commit。本报告及所有产物均无密钥。
