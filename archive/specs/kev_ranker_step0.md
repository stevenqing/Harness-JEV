# Kev Ranker — Step 0 Spec（离线 head 训练 + flipped/regressed 预测验证）

> 给 coding agent 的可执行 spec。目标：只用**现有进化日志**（零新 rollout）验证「kev 能否学会预测某个
> 候选改动会不会让某个 task 翻盘（flip）/回退（regress）」。赢了才做 Step 1（接进进化循环）。
> 安全红线：token/key/secret 不进 report/log/commit。

---

## 1. 目标与验收门槛

**验证一句话**：给定「task t 在 config Ci 下的失败轨迹」+「Ci→Cj 的改动 diff」，kev 能否预测 t 在 Cj 下会不会翻盘。

**验收门槛（Go/No-Go）**：在 **held-out task** 与 **held-out run** 两种切分上，kev 对 flipped/regressed 的
二分类 AUC 必须 **同时** 超过下面 3 个 baseline：

1. **base-rate**（照抄旧结果）：flip 的先验概率。
2. **predicted_affected**（meta 写的）：candidates.md 里的 `Tasks affected` 列表。
3. **random**：0.5。

并且**捷径检查**（§5.3）必须证明 kev 在读「轨迹 × diff」的交互，而不是格式/base-rate。任一捷径检查里
kev 掉到 baseline 水平 → 视为「没在真读轨迹」，同样判 No-Go。

---

## 2. 关键前提（已摸清，coding agent 直接引用，不要重查）

### 2.1 kev 训练接口（`/mnt/llmshared-ssd-hd/shishuqing/kev/`）

- 训练入口：`kev/.venv/bin/python -m kev.train --data <jsonl> --init_from <ckpt> ...`
- 架构：`DecisionModel` = 冻结骨干（Qwen3.5-4B-Base，可加 LoRA）+ `PointerHead(d→256)`。
  `<decide>` token 的 hidden 对每个 option 的 `</opt>` hidden 做 pointer readout → K 维 logits。
- **软目标已原生支持**：JSONL 里 question 带 `"target": {"opt_a":0.7,"opt_b":0.3}`（而非 `label`），
  `question_loss` 走 `-(target · log_softmax(z))` —— 这就是 RLCD/蒸馏的 loss，第 0 步直接用它做「候选排序」软标签。
- **anchor 正则已原生支持**：`--anchor <json>` + `--anchor_w <w>`，loss 为 `KL(anchor || model)`。
  用户说的「加 L2 把 head 往 kev 初始读法拉」即此（KL 而非 L2，语义一致）。anchor 应指向**当前部署的 kev**（见下），不是 base model。
- 部署 kev checkpoint：`/mnt/llmshared-ssd-hd/shishuqing/models/kev-4b`（serve 用 `--run models/kev-4b`；
  base 已改成本地 `models/Qwen3.5-4B-Base`，见 `kev_prep.py`）。第 0 步 `--init_from` 用它。
- 数据格式（`kev.data.load_records`，`--data` JSONL 一行一 record）：
  ```json
  {"state": "<轨迹渲染字符串>",
   "questions": {"q0": {"type": "noul", "instructions": "...", "criteria": {"true":"...","false":"..."}, "label": true}}}
  ```
  `type` ∈ `noul` / `choice` / `score`；`label`：noul→bool，choice→option key，score→level index；
  `target`（可选）→ 软目标（keyed by option name，自动归一化）。

### 2.2 现有日志结构（`Harness-JEV/recipe/agent_evolver/runs/evolve/*_heldin/`）

每个 heldin run 有 `R0..R5` 轮，每轮目录：
- `config.yaml`：HarnessConfig（guidance 在 `TextEvolverSpecProcessor` 的 `guidance` 字段；system prompt 在 `SystemPromptProcessor`）。
- `trajectories/*.md`：**per-task 结果在这里**，frontmatter 含 `task_id, goal, exit_reason, steps, invalid_actions, reward`（`reward=1.0` 过 / `0.0` 失败）。
- `evolve/_meta_scratch/candidates.md`：meta 写的候选，每候选含 `[lens|lever|intent]` 三轴 + `Tasks affected: 3,10,18,...`（= predicted_affected）+ 自然语言 diff 描述。
- `comparison.json`：**只有聚合**（`rounds[].{round,config,tasks,passed,pass_rate,mean_reward}`），**无 per-task、无 Δ、无选择概率**。

**三个已确认的缺口**（对应方案「还没解决的四件事」前两件）：
1. 现有日志**无选择概率** → 只能做有偏训练 + held-out 评估，不能做无偏 IPS（IPS 留到 Step 1 产生的新日志）。
2. 现有 run **每轮只有一个真正 apply 的 config**（candidates.md 有多个候选但最终只落一个）→ 排序样本只能靠「任意两轮配对」+ 多臂 R0 分叉凑，第 0 步排序检验有限，主线是 pass_prob。

---

## 3. 数据构建

### 3.1 Run 筛选规则（`benchmark=alfworld`, `num_tasks=64`, `seed=1234`, `agent_model=Qwen3.5-4B`）

纳入（预期 ~15-17 个，coding agent 先跑清单审计确认）：
- `ov3x_{llm,prior,enforce,meta}_heldin`
- `ov3xn_{llm,prior}_heldin`
- `ov3xab_{lensonly,fixedpred,metascorer}_heldin`
- `ov3xcp_0.5_heldin`（constant-prior，若已出 R0-R5）
- `quality_{llm,kev}_heldin`、`ov_q0_{llm,kev}_heldin`、`q0_llm_heldin`、`a3_llm_s1234_heldin`

**排除**：
- `smoke_*`（num_tasks 少）。
- `*_2b` / `*_0.8b`（不同底座，轨迹分布不同，不能和 4B 混）。
- `tj-*`（多 seed，task 集因 `game_idx` shuffle 不同，flip/regress 无法跨 seed 对齐）。
- `sw*`（WebShop）、`tc*`（TextCraft）—— Step 2 才做，本步不纳入。

**审计产物**：`data/run_manifest.json`（每 run：tag、benchmark、agent_model、seed、num_tasks、每轮 `passed/tasks`、纳入/排除及原因）。这是后续一切数据构建的底账。

### 3.2 per-task 结果提取

对每个纳入 run 的每轮 Rk，读 `Rk/trajectories/*.md` frontmatter 的 `task_id` 与 `reward`，得到
`pass[k][task_id] ∈ {0,1}`。以 `task_id` 为对齐键（同一 run 内 seed 固定 → task_id 语义一致）。

### 3.3 候选 diff 的文本表示

第 0 步的「候选改动」= **相邻两轮 guidance 的差异**（不引入新 rollout，也不依赖 meta 是否写了候选）：
- 从 `config.yaml` 提取每轮 `guidance` 全文（无 guidance 的轮视为空/基线）。
- `diff(Ri → Rj) = difflib.unified_diff(guidance(Ri), guidance(Rj))` 的紧凑文本：只保留 `+`/`-` 行，去
  `@@` 头，逐行 `+`/`-` 前缀，截断到 **≤ 512 字符**（超长取前 512 并加 `…[truncated]`）。
- 相邻轮配对为主 `(Ri, Ri+1)`；数据不足时允许 `(Ri, Rj)` 任意 `i<j`（在样本 `_meta` 里记 `src_round`、`dst_round`）。

### 3.4 样本构造（noul，pass_prob —— 第 0 步主线）

对每个纳入 run、每对 `(Ri, Rj)`（默认相邻）、每个 `task_id t`：

- **state** = `t` 在 `Ri` 的轨迹渲染（固定格式，见 §3.5）。
- **question（noul）**：
  - `instructions` = `"Below is a failed trajectory and a proposed change to the guidance. Will this change make this task pass?"`
  - `criteria.true` = `"the task flips from fail to pass under the change"`
  - `criteria.false` = `"the task stays failing, or a passing task regresses to failing"`
  - 关键：**diff 文本放进 `instructions`**（拼在 change 后），**不是** state。state 只放轨迹。这样 kev 的
    `<decide>` 同时 attends 轨迹（state）与改动（question branch），符合 pointer readout 的「读轨迹 × 读候选」。
- **label**：
  - `t` 在 `Ri` 失败 且 `Rj` 成功 → **flipped**，label=`true`。
  - `t` 在 `Ri` 成功 且 `Rj` 失败 → **regressed**，label=`false`。
  - 两轮结果相同（都失败/都成功）→ **不进训练**（信号弱，先不要）；但另存一份 `unchanged.jsonl` 供消融备用。
- **`_meta`**（每样本，用于切分/溯源，不进入 kev 的 state/questions）：`run_tag, task_id, src_round, dst_round, src_pass, dst_pass, kind(flipped|regressed), diff_sha256, state_sha256`。

数据规模预期：~16 run × 5 个相邻配对 × 64 task × (flip+regress 占比，历史 ~10-20%) ≈ **1000–2000 样本**。
若 <300 样本，把「任意两轮配对」开满并上报（说明需要更多 run 或 Step 1 产数据）。

### 3.5 state 渲染格式（固定，不用 agent prompt）

复用 Harness-JEV 已验证的 compact 渲染（`recipe/agent_evolver/decision_prior.py::_render_state` 的语义）：
```
goal: <goal>
exit_reason: <exit_reason>  steps: <steps>  invalid_actions: <invalid_actions>
actions: <a1>; <a2 [invalid]>; ...
```
动作序列截断到 ≤ 900 字符（`…` 结尾）。**同一 task 在不同轮用同一渲染器**，保证 kev 的输入不随 harness prompt 变而变。

---

## 4. 训练

### 4.1 切分（两种都要做，各一份报告）

- **held-out task**：按 `task_id` 切（例如 80/20，seed 固定）。同 run 的轨迹可能跨 train/eval task，但 label 由
  「改动的因果」定义，task 级切分测的是「对新 task 的翻盘预测泛化」。
- **held-out run**：按 `run_tag` 切（例如留 3 个 run 不训）。测「对新 run 的泛化」（更强，接近 Step 1 真实场景）。

### 4.2 anchor（把 head 拉回部署 kev）

1. 用部署 kev（`models/kev-4b`）在**全部训练+评估样本**上跑一遍（`kev.benchmark` 或 `LocalPredictor`），
   得到每个 record 的 zero-shot 概率分布，导出为 `--anchor` 的 JSON：`{"targets": {"<record_id>": {"q0": {"true": p, "false": 1-p}}}}`。
   record_id 用样本 `_meta` 里的稳定 id（如 `run_tag/task_id/src_round→dst_round`）。
2. `--anchor_w` 从 `{0.1, 0.5, 1.0}` 里扫（小网格），选 held-out 上 AUC 最优。

### 4.3 训练命令（最小改动，复用现有管线）

```bash
cd /mnt/llmshared-ssd-hd/shishuqing/kev
.venv/bin/python -m kev.train \
  --data <built>.jsonl \
  --init_from /mnt/llmshared-ssd-hd/shishuqing/models/kev-4b \
  --anchor <anchor>.json --anchor_w <w> \
  --lora 16 --lr 2e-5 --head_lr 2e-4 \
  --epochs 1 --batch 1 --accum 8 --seed 0 \
  --out runs/kev-ranker-s0/<variant>
```

- `--head_lr > --lr`：pointer head 从 scratch 训，LoRA 小步微调。
- 第 0 步用**端到端 LoRA+head**（现有管线，最小改动，单卡 ~1-2h）。「缓存 hidden 只训 head」是 Step 1 在线
  更新的加速项，**本步不做**（除非端到端超预算）。
- 记录 `training_metrics.json`（kev.train 自动写）里的 `wall_seconds / records_seen / optimizer_steps`。

### 4.4 消融（至少跑这些，决定哪个成分有用）

| variant | 说明 |
|---|---|
| `full` | 全量样本 + anchor |
| `no-anchor` | `--anchor_w 0` |
| `flip-only` | 只 train flipped（regressed 留 eval），看 regressed 泛化 |
| `regress-only` | 只 train regressed |
| `hard-label` | 同样本但 `target` 换成 hard label（对照软标签是否必要，本步 pass_prob 是 noul，此项留给排序消融，见 §6） |

---

## 5. 评估

### 5.1 指标（对 held-out task 与 held-out run 各算一套）

- **AUC**（flip=true vs regress=false 的二分类，用预测的 `p(true)`）—— 主指标。
- 附加：`top-1 acc`（p>0.5 判 flip 的准确率）、`flipped 召回 @ 0.8 precision`、样本数、flip 占比（base-rate）。
- 分桶报：`flipped` 与 `regressed` 两类的 AUC 单独报（别只报混合）。

### 5.2 三个 baseline（必须同切分同指标）

1. **base-rate**：flip 先验 `P(flip)`（在对应切分的训练集上估计，eval 集上评）。等价于常数预测。
2. **predicted_affected**：从 `evolve/_meta_scratch/candidates.md` 解析每个候选的 `Tasks affected` 列表；
   task 在列表内 → 判 flip。注意：candidates 描述的是「下一轮」的预期，需按 `dst_round` 对齐到对应候选；
   无法对齐的样本从该 baseline 里剔除并记 `n_skipped`。
3. **random**：0.5。

### 5.3 捷径检查（三组，证明在真读轨迹）

对训练好的 `full` 模型，构造三组对照数据、分别重训（或只 eval）并报 AUC：

1. **diff-only**：state 置空（只留 diff 在 instructions）→ 若 AUC 不掉，说明信号全在 diff 文本。
2. **state-only**：instructions 去掉 diff（只留轨迹）→ 若 AUC 不掉，说明信号全在轨迹 base-rate。
3. **shuffled-diff**：把 diff 随机错配到轨迹（state 与 diff 无真实因果）→ 若 AUC 仍高，说明在学 base-rate/格式，不是交互。

判定：三组都显著低于 `full`（且接近对应 baseline）→ 通过（kev 在真读「轨迹 × 改动」）。

### 5.4 报告产物

`report.md`（或 JSON）必须含：run 清单、样本规模、切分方式、每个 variant × 每个切分的 AUC 表、3 baseline 对比、
3 捷径检查对比、anchor_w 扫描、训练 wall-clock、以及**明确的 Go/No-Go 结论**。

---

## 6. 排序（有限附加，非验收主线）

现有日志无多候选，只能做 pairwise 有限检验，**放在 pass_prob 之后**，不阻塞 Go/No-Go：

- 构造：同一 `task_id t`、同一 `Ri`，取两个不同 `Rj1, Rj2`（各自 diff），choice 问题「which change more likely
  flips this task?」，`label` = 实测 flip 的那个（都 flip/都不 flip 则跳过或 softmax 目标）。
- 用 `--data` 的 `target`（软目标 = 两候选 Δ 的 softmax）训一个 choice 版，报 top-1 acc vs random（1/K）。
- 若 pairwise 样本 < 100，只做 hard-label 消融不追求性能，明确标注「排序验证留 Step 1」。

---

## 7. 交付物清单

1. `data/run_manifest.json`（run 审计）
2. `data/build_step0.py`（日志 → JSONL + anchor + 切分；幂等，输入 run 列表，输出 train/eval JSONL 与 `_meta`）
3. `data/anchor_build.py`（部署 kev → `--anchor` JSON）
4. `data/eval_step0.py`（训练产物 → AUC / 3 baseline / 3 捷径检查 / 报告）
5. `data/run_train.sh`（训练 + 消融 + 扫描的一键脚本）
6. `data/report.md`（§5.4）
7. 训练 checkpoint（`kev/runs/kev-ranker-s0/<variant>/`）

**验收**：跑 `data/run_train.sh` 后 `data/report.md` 自动生成并给出 Go/No-Go。全程零 rollout、零 meta 调用。

---

## 8. 安全红线

- 训练/评估脚本、JSONL、report、commit 一律不含 token/key/secret。
- meta 的 candidates.md 只取 `Tasks affected` 与 diff 文本用于 baseline，不落任何凭证。
- 输出 JSONL 的 `state`/`instructions` 是轨迹渲染与 guidance diff，不包含 agent 运行时凭证。
