# Harness-JEV 全结果汇总（ALFWorld / WebShop / FrozenLake / Sokoban）

> 项目主线：**只进化 harness（system_prompt / guidance / history_window / processors），模型权重永不训练**。
> agent 模型 = Qwen3.5-4B（冻结，vLLM 服务），meta-agent = deepseek-v4-pro（mioffice anthropic SDK）。
> 更新日期：2026-10-07。

---

## 0. 方法与口径（所有表都以此为准）

| 项 | 值 |
|---|---|
| agent 底座 | Qwen3.5-4B（冻结；模型矩阵里另测 2B / 0.8B） |
| meta 后端 | deepseek-v4-pro（文本协议进化器 `recipe/agent_evolver/`） |
| 采样 | 早期 temp 0.4（±3–6pt 噪声）；三轴 / kev 先验用 temp=0 贪心确定性 |
| 评测协议 | **held-in 进化 / held-out 只测**（真 OOD 泛化探针） |
| held-in/out 切分 | ALF：in_distribution(140) vs out_of_distribution(134)；WS：官方 train(goals 1500+) vs test(goals 0–499)；FL/SK：evolve split vs test split（各 64 level，4 tier × 16） |
| 三轴 | `llm`（纯 meta）/ `prior`（kev 软先验注入，deepseek 可覆盖）/ `enforce`（kev 硬绑定杠杆 + candidate retrocheck gate） |
| kev | Qwen3.5-4B-Base 冻结底座 + LoRA(r=16) + pointer head —— 训练产物（非 harness-only），作「打分头上界」与软/硬先验源 |

---

## 1. 基线锚点（Qwen3.5-4B 裸基线，未进化）

| benchmark | split | success | mean_reward |
|---|---|---|---|
| ALFWorld | held-in (in-distribution 64) | **0.344** | — |
| ALFWorld | held-out (OOD 64) | **0.344** | — |
| WebShop | test (human 64) | 0.281 | 0.535 |

> held-in == held-out（都是 0.344）→ 裸模板无 held-in 过拟合，是干净进化起点。

---

## 2. 纯 harness 进化（llm meta，无 kev）

### 2.1 ALFWorld —— 大赢，meta 自摸动作语法

| 阶段 | held-in | held-out |
|---|---|---|
| 裸基线 | 0.359 | 0.281 |
| v1-R2（3 条 state-tracking 规则 + history_n 2→8） | 0.453 | 0.609 |
| v2-R2（扩到 5 条，**精确动作词表**） | **0.703** | **0.750** |

- **弧线**：held-in 0.359→0.703、held-out 0.281→0.750 —— 两 split 都约翻倍，且 held-out ≥ held-in（**零过拟合**）。
- **本质**：meta 从失败轨迹里**独立摸出 ALFWorld 的精确动作语法**（`take X from Y` / `heat X with microwave` / `cool X with fridge` / `clean X with sinkbasin` / 先 go to 再 move），治好了 malformed action 失败主因。系统 prompt 始终未动，只改 `guidance:` + `history_n`。

### 2.2 WebShop —— 最难的迁移

| 场景 | 结果 |
|---|---|
| 人类池内（task 0–63 进化+测，无 split） | R0 0.219/0.435 → R2 0.359/0.635（**同池假象**） |
| synthetic=held-in 进化 → human=held-out 测 | **零迁移**（mean +3.1pt，落噪声带） |
| 官方 train → test | **弱正**：128-task matched A/B 基线 0.211/0.494 → 0.242/0.568（**mean +7.4pt**，pass +3.1pt 不显著） |

- meta 摸到「商品页 click 每个属性选项 + 短 query + 买 best」的评分语法，人类池内分涨。
- 但 synthetic 模板目标的语法**不迁移**到人类自然语言目标（真 OOD 判负）；官方 train→test 只剩 **mean-only 弱正**（离满分更近，没多解出单）。
- **与 ALF 分野**：ALF 的动作词表是任务无关的、真泛化；WS 的评分语法是池相关的、跨池衰减。

### 2.3 FrozenLake —— decoding 是前提，prompt 措辞是杠杆

| 设置 | 结果 |
|---|---|
| L4, thinking **off** + max_tokens 32 | 3.9% → 平台 7–8%（退化，decode 瓶颈） |
| L4, thinking **on** + max_tokens 4096 | 基线直接 **56.2%**（×18，真寻路） |
| **L16, thinking off + max_tokens 4096** | 53.9% → **68.9%** held-in，**65.0%** held-out（gap 3.9pp） |

- **根因确认**：早期 FL 全线坍缩是 **decoding 伪影**（thinking off + 短预算 → 模型全吐 `[up]`/`[down]`），不是 prompt/能力。
- **L16 进化赢家 harness（R5）三招**：① 列坐标双向核对（治读盘数错列）；② 防死循环逃逸（Last moves 来回弹同一对格子=口袋 → 走一步暂时增距的 SAFE 步）；③ 只看 P 的 4 邻居 + 强制 `<action>` 收尾（压缩全盘扫描）。全部是 prompt 措辞层面的真实失败模式杠杆，泛化 gap 仅 3.9pp。

### 2.4 Sokoban —— 尚未攻破

| 设置 | 结果 |
|---|---|
| L16 基线（thinking off + 4096） | **21.1%**，invalid **64.7%** |
| ragen 协议（`<think>/<answer>` + thinking off） | **0%**，parse_fail **85%** |

- Sokoban 比 FL L16 硬得多（大棋盘 + 冗长推理更易截断）；bracket 三轴 run 中途被 kill、无干净结果。**目前无成功解法**。

---

## 3. 模型规模矩阵（4B / 2B / 0.8B，纯 harness 进化 6 轮）

| 模型 | ALF held-in 基线→best | ALF held-out | WS test (mean) |
|---|---|---|---|
| 4B | 0.484 → **0.703** | 0.750 | 0.242 (0.568) |
| 2B | 0.109 → **0.250** | 0.219 | 0.297 (0.656) |
| 0.8B | 0.000 → **0.078** | 0.000 | 0.109 (0.343) |

三条规律：
1. **绝对分数随规模单调坍缩**（ALF best 0.703 / 0.250 / 0.078）。
2. **二值 vs 连续 reward 决定「能力地板」是否致命**：ALF（二值）0.8B 卡 0 且 held-out 归零；WS（连续）0.8B 基线 mean 0.016 → best 0.394（×24），test 保留 87% —— 只要有部分分可拿，harness 进化在地板模型上仍有效。
3. **相对增益随规模下降而升、绝对反之**（WS mean Δ 4B +8.4 / 2B +9.5 / 0.8B +37.8pt）。

---

## 4. kev 先验 + 三轴（llm / prior / enforce）

### 4.1 kev Mode A（软先验，temp=0 干净 A/B，ALFWorld）

| 指标 | llm arm | kev arm |
|---|---|---|
| held-in 终局 | 0.766 | **0.906**（+14pt） |
| held-out 探针 | 0.797 | **0.969**（+17pt） |

- kev 从 R1 起每轮单调领先，held-out(0.969) > held-in(0.906) → 无过拟合，先验不是背 held-in 题。

### 4.2 三轴总表（held-out 探针）

| benchmark | baseline | llm | prior | enforce | 冠军 |
|---|---|---|---|---|---|
| ALFWorld (pass) | 0.297 | 0.781 | **0.844** | 0.594 | **prior** |
| WebShop (pass / reward) | — | 0.312 / 0.560 | **0.344 / 0.630** | 0.203 / 0.567 | **prior** |
| FrozenLake bracket (pass / reward) | 0.078 | **0.672 / 0.690** | 0.531 / 0.551 | 0.531 / 0.557 | **llm** |
| FrozenLake ragen (pass / reward) | 0.016 | 0.094 / 0.129 | **0.234 / 0.282** | 0.125 / 0.163 | **prior** |

（ALF/WS/FL-ragen 是 temp=0 三轴；FL-bracket 是 thinking-on 三轴；各臂 held-out 用 held-in 最优 config 重放。）

**核心规律**：
1. **soft prior > llm > hard enforce** 在 ALF / WS / FL-ragen 上**一致复现** —— kev 软先验给 deepseek 保留覆盖权，硬 gate 把能过的 candidate 也 reject、还砍伤探索空间。
2. **FL-bracket 反转（llm 赢）**：thinking-on 高基线（0.078→0.719）下，纯 llm meta 已经很强，kev 先验反而**拖累泛化**（prior held-out 0.531 比 held-in 0.688 掉 15.7pp，是过拟合信号）。→ **kev 先验的增益在有信号的低基线场景（ALF 闭集失败密集 / FL-ragen 弱解码）里最显著，高基线场景是噪音**。
3. **enforce 硬 gate 全程最差且震荡**（FL-bracket R4 曾塌到 0 pass）：lever 4-way（configuration/control/action/instruction）对裸文本 loop 粒度太粗，kev 每轮 lever 恒判 `instruction`（conf 0.24–0.26），硬绑定是噪音；candidate retrocheck（`intent=corrective pass_prob 0.82–0.98`）全程通过、无区分力。

---

## 5. 参照系：系统-1 打分方法（非 harness 进化，仅基线标定）

| 方法 | ALF seen | ALF unseen | WS test (success / reward) |
|---|---|---|---|
| gen（生成式基线） | **0.350** | **0.321** | 0.182 / 0.444 |
| semif（直接 logit） | 0.114 | 0.104 | 0.182 / 0.615 |
| kev（指针头+LoRA） | 0.164 | 0.172 | **0.252 / 0.627** |
| djev（Diffusion） | 0.150 | 0.082 | 0.224 / 0.613 |
| laya（encoder+决策头） | 0.021 | 0.007 | 0.080 / 0.310 |
| verdict2（GLiClass） | 0.036 | 0.030 | 0.120 / 0.427 |

- 生成式 `gen` 在 ALF 二值任务上仍是自回归最强（0.350）；非自回归打分方法全线垫底。kev 在 WS 连续 reward 上六方法第一，标定「训练打分头」上界 —— harness-only 进化碰不到这个上界，只能朝 semif→kev 方向做不碰权重的逼近。

---

## 6. 关键结论（一句话版）

1. **纯 harness 进化真有效**：ALF held-in 0.359→0.703、held-out 0.281→0.750（+2x），FL L16 53.9%→68.9%（held-out 65%），都**无过拟合**。
2. **赢的本质 = meta 发现该 benchmark 的「评分/动作语法」**：ALF=精确动作词表、WS=属性点击、FL=坐标核验+破环；系统 prompt 全程未动，只改 guidance。
3. **decoding 是 gridgames 的前提**（thinking on 才能寻路），prompt 措辞是后续杠杆。
4. **kev 软先验是通用增益**（ALF/WS/FL-ragen 三轴全赢），但**高基线场景失效**（FL-bracket llm 赢）；**硬 enforce 恒为负**。
5. **WebShop 迁移最难**（synthetic→human 零迁移、train→test 仅 mean 弱正），**Sokoban 仍未攻破**。
6. **模型规模**：绝对分随规模坍缩，连续 reward 任务保留相对增益、二值任务存在硬地板。

---

## 附：产物位置

- 结果 run 目录：`recipe/agent_evolver/runs/evolve_gridgames/`（gridgames）、`runs/evolve/`（ALF/WS 早期）、`runs/agent_evolver/official/`（系统-1 方法）。
- 分项详档：`docs/gridgames-fl16-evolution-results.md`（FL L16）、`docs/gridgames-difficulty-map.md`（难度背景）。
- 记忆（本机 `.claude/projects/-root/memory/`）：`kev-three-axis-result`（ALF 三轴）、`harnessx-jev-webshop-3axis-result`（WS 三轴）、`ragen-gridgames-3axis-result`（FL-ragen 三轴）、`kev-prior-mode-a-result`（kev Mode A）、`harnessx-jev-model-matrix`（规模矩阵）、`gridgames-thinking-decoding-win`（decoding 根因）。
