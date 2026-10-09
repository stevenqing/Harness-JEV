# Visibility story：kev 当 reviewer（ALF/WS）+ readout agent（gridgames）

> Coding spec 的执行计划。写在 2026-10-07，rollout 开始前。
> 全程不训练/不改任何模型权重。kev 原样用，semif 是冻结 readout。
> 配套规格书（原始 spec）：`## Next tasks. Visibility story: …`（本文件是其落地版 + 差距盘点）。

---

## 0. 两条主张（判断标准先立起来）

- **Claim A（reviewer 有价值）**：失败模式只在轨迹语义里可见（ALFWorld/WebShop）时，一次过场的判断模型打失败标签，给 meta-agent 一个它本来只能靠猜的信号；失败模式从 state 可读（gridgames）时，同一个 tagger 相比规则零增量。
- **Claim B（readout 能到 gen 到不了的 harness）**：当打标签没用、因为瓶颈是模型自己的逐步决策（Sokoban），用**同一套冻结权重当 readout**，让 harness 每步看到一个概率向量，进化能到达 outcome 反馈的 gen agent 到不了的 harness。

Track A 测 Claim A，Track B 测 Claim B，Track C 只在两 gate 都过才跑。

---

## 1. 基座修正（S1–S4，先做）

- **S1** 本 spec 里所有生成式 run 统一解码：thinking on、`max_tokens` 4096、单一 temperature，写进报告。早期 thinking-off run 不复用。
- **S2** gridgames 用 bundle `gridgames/` 严格按 SPEC：3584-level manifest、per-cell 切分 evolve 256 / gate 64 / test 128、episode contract、gate G0+G1。退役 64-level ad-hoc 切分。bundle 的 R0 解码改成 S1。bundle 预测 P1–P6 作废，用 §5 替换。
- **S3** 每个报告数字 3 seeds + 95% 分位 bootstrap（over levels/tasks，10k 次）。差值区间含 0 才算非真差异。
- **S4** 每个进化出的 harness 也被 null 模型（gridgames=uniform 随机走；ALF/WS=uniform 随机合法动作）在同一 split 玩一遍，报告里放一列挨着模型列。

---

## 2. Track A（reviewer 价值）—— gate GA

- **A1** 给现有 held-in run 的每条失败轨迹建独立失败标签（ALFWorld 从 expert plan：`find`=从未取得目标物 / `acquire`=取得但没完成下一步 expert 动作 / `other`；gridgames 从 oracle：`fell_in_hole`/`dead_push`/`loop`/`budget`）。逐轨迹比较 kev lens tag vs 标签。报告准确率、Cohen's κ、kev lens 分对 `find` 的 AUC。
- **A2** 规则 tagger：从轨迹文本确定性打同一 tag 集（例：`find`=没有成功 `take`）。对同样标签报告准确率 + κ，ALFWorld 和 gridgames 都做。
- **gate GA**：ALFWorld 上 kev 逐轨迹准确率 ≥ 规则 +0.10 且 κ ≥ 0.4；gridgames 上两者差 ≤0.03。GA 不过 → Track A 停、reviewer claim 撤出 paper。两种结果都报告。
- **A3** 三臂（`llm`/`kev`/`rule`，3 seeds，held-in 进化 6 轮 / held-out 测，S1 解码，**不跑 enforce**）。`kev`=现有 prior 注入；`rule`=同一 brief 格式注入规则 tagger。报告 held-out success+reward 带区间，加 S4 null 列。

---

## 3. Track B（readout agent）—— gate GB

- **B1** readout R0 无进化：semif + kev 当 agent，8 格 gate split、每 level 4 episodes。题=当前棋盘 + 四动作选项，harness=bundle R0 renderer+prompt。报告 `success − floors.gate.uniform` per cell 带区间。readout 曲线起点。
- **B2** readout 进化：FL-L16 + SK-L8，meta 对 readout agent，6 轮 3 seeds，split 按 S2。可进化 harness = HarnessSpec v2（state renderer / question wording / option grouping 含 two-level choice / tried-action mask / loop breaker，来自 `harnessx_jev_semif_spec.md`）。meta 读的轨迹带每步概率向量。gen agent（S1 解码）同 cell 同 meta 同 seeds 作对照曲线。bundle SPEC §6.3 适用：不超一步搜索、不 import bundle、无 level-specific 内容。
- **gate GB**：SK-L8 gate split 上 best evolved readout 的 `success − floors.gate.uniform` ≥ 0.10 且区间不含 0，且 best evolved gen 同 split 不达标。GB 因"都不出 floor"失败 → Sokoban claim 撤出、Track B 只报 FL。GB 因"gen 也出 floor"失败 → Claim B 错、报告直说。
- **B3** test split 一次性，B2 每格每臂 best harness，每 level 4 episodes，带 null 列。

---

## 4. Track C（GA 和 GB 都过才跑）

- **C1** 把 A 的 kev reviewer 插进 B2 的 SK-L8 loop，画 `success vs cumulative rollouts / meta tokens`，三臂 `llm`/`kev`/`rule`。Claim A 预测这里无差异——是校验，不是头条。

---

## 5. 预测（rollout 前冻结，2026-10-07）

| # | 预测 | 判错条件 |
|---|---|---|
| P1 | ALFWorld 上 kev lens 准确率 ≥ 规则 tagger +0.10 | gap < 0.10 |
| P2 | gridgames 上 kev lens 准确率与规则差 ≤ 0.03 | gap > 0.03 |
| P3 | ALFWorld held-out：`kev` ≥ `rule` +0.05 success | 未达 |
| P4 | WebShop held-out：`kev` ≥ `rule` +0.03 mean reward | 未达 |
| P5 | Sokoban L8 readout R0（无进化）在 uniform floor ±0.05 内 | 高于 floor+0.05 |
| P6 | Sokoban L8 evolved readout 出 floor ≥ +0.10 | 未达 |
| P7 | Sokoban L8 evolved gen 不出 floor +0.10 | 出了 |
| P8 | FL-L16 evolved readout 在 evolved gen ±0.05 内 | 一方超另一方 >0.05 |
| P9 | C1 三臂在等 rollout 下两两 ≤0.05 | 任一对 >0.05 |

---

## 6. 报告与运维

- 产出 `REPORT_visibility.md`：bundle S2 后 G0/G1 输出；解码设置 + 每 env 一个完整 prompt；A1/A2 逐 tag 混淆矩阵；A3 区间+null 列；B1 per cell；B2 每 seed 曲线；B3 test+null；每条预测判决+背后数字；专节 "What the data do not support"；所有偏离 spec 记录。
- 每 episode 一条 JSON：environment / arm / seed / round / level-or-task / episode-index / `report()` 字段 / **每步 raw 输出或概率向量 + parsed action**。所有数字能从该文件 + bundle `levels.jsonl`/`tiers.json` 重算，附重算脚本。
- 运维红线：episode 结束即写盘、重启跳过已完成；infra 失败同 seeds 重试 ≤3 次、永不计为模型失败；>5% episode 弃跑 → 该 cell/task 整跑。
- 安全：token/key/secret 不进 report/log/commit。

---

## 7. 顺序与停止规则

`S1–S4` → `A1+A2` 与 `B1` 并行（无资源冲突）→ `GA` 决定 A3 → `B2` 与 `A3` 并行 → `GB` 决定 C1。任何 gate 不过 → 停并报告，不加臂。

---

## 8. 差距盘点（已核对磁盘）

| 类别 | 已存在（复用） | 从零建 |
|---|---|---|
| 基座 | bundle(`levels.jsonl` 3584 / `tiers.json` floors.gate.uniform / `floors.jsonl` / `verify_levels.py` G0 / `reference_run.py` G1a-d / `stubs.py` / `r0.py` / `make_levels.py` SPLIT_SIZES)、kev 三轴 prior/enforce、`harnessx/decision`(KevBackend) | bootstrap 报告层、null live 列、ALF/WS 的 S1 解码翻转 |
| Track A | kev prior 注入(`run.py --decision-backend`) | 独立失败标签、规则 tagger、`rule` 注入臂 |
| Track B | kev 后端 | **SemifBackend、readout agent loop、HarnessSpec v2（two-level choice/mask/loop-breaker，`harnessx_jev_semif_spec.md`）** |
| 报告 | — | `REPORT_visibility.md` + episode JSONL + 重算脚本 |
