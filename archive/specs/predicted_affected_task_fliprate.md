# SPEC：用 task 历史翻转率替代 `predicted_affected`（journal 归因降噪）

> 给 coding agent 的可执行 spec。**零训练、零新 rollout**，纯用现有日志的统计量替代进化循环里一处噪声字段。
> 安全红线：token/key/secret 不进 report/log/commit。

---

## 1. 背景：为什么做（已验证的信号）

kev ranker Step 0b（`data/kev_ranker_s0b/`）的模型阶梯给出一个硬结论：

| 模型 | 是什么 | 翻盘预测 AUC（ALF / WS） |
|---|---|---|
| **B1 = task 历史翻转率** | 每个失败 task 在历史 run 里 fail→pass 的频率，当「这次会不会翻盘」的预测 | **0.747 / 0.810** |
| **B2 = `predicted_affected`** | meta 写候选时声明的「预期受影响 task」列表 | 0.509 / 0.555（≈随机） |
| L1 = B1 + diff 特征 | B1 再加改动字符数/字段数/规则数 | 0.747 / 0.810（**与 B1 逐位相等**） |

三个事实：①「哪个 task 会翻盘」高度可预测；② 预测力 100% 来自 task 本身（历史翻转率），**加 diff 特征零增量**；③ meta 的 `predicted_affected` 是纯噪声，比简单统计差 +0.235/+0.249 AUC（CI 显著）。

**结论**：`predicted_affected` 这个字段现在携带的是噪声。用「task 历史翻转率 top-k」替代它，能让 journal 归因从「随机」变成「准」。

**边界（必须诚实）**：这是 **task 级**信号（预测「哪个 task 翻盘」），**不是 candidate 级**（预测「哪个候选更好」）。它不复活 kev ranker（那个要 diff 侧信号，Step 0b 前提二已 FAIL）。本改动只做「归因降噪 + 注意力引导」。

---

## 2. 现状：`predicted_affected` 在哪里被消费

`harnessx/meta_harness/journal.py`：
- `Entry.predicted_affected`（`journal.py:106`）从 journal frontmatter 读。
- `compute_attribution(predicted, passed_now, passed_before, ...)`（`journal.py:605`）把 predicted 里每个 task 分类成 `flipped` / `still_F` / `regressed` / `still_T` / `absent`。

`recipe/agent_evolver/run.py`（journal back-fill，R1+，`run.py:460-495`）：
```python
attribution = _journal.compute_attribution(
    entry.predicted_affected, passed_now=cur_passed, passed_before=prev_passed,
    appeared_now=cur_appeared, appeared_before=prev_appeared,
)
predicted_set = set(entry.predicted_affected)
regressed_unpredicted = sorted(
    (prev_passed & prev_appeared & cur_appeared) - cur_passed - predicted_set
)
```

`regressed_unpredicted` 语义 = 「回退但 meta 没预测到」的 task，写进 journal 的 `regressed_unpredicted` frontmatter，下一轮 meta 读它做回归风险判断。

---

## 3. 改动点

### 3.1 核心（必做）：journal back-fill 用翻转率替代 meta 的 predicted

**只改 `run.py` 一处**（`run.py:478-485` 的 `entry.predicted_affected` → `fliprate_predicted`），`compute_attribution` 和 `regressed_unpredicted` 的调用签名不变。

新逻辑：
1. 读一张预计算的静态表 `data/task_fliprate.json`：`{task_id: flip_rate ∈ [0,1]}`。flip_rate 定义 = 该 task 在**所有纳入 run 的全部相邻轮对**里「src 失败 → dst 通过」的翻转次数 ÷ 「src 失败」的次数（即 B1 的精确口径，Step 0b 已算，从 `real_pairs.jsonl` + `cells` 可重算）。无历史 → 填 stratum 均值。
2. 每轮 back-fill 时，`predicted_affected` 不再是 meta 的 frontmatter，而是：
   - `fliprate_predicted = 当前失败 task 里 flip_rate 最高的 k 个 task_id`（k 取 meta 原 predicted 的长度，保口径；meta 没写则默认 k=8）。
   - 只纳入「本轮确实失败」的 task（否则没有 flip 可预测）。
3. 把 `fliprate_predicted` 传给 `compute_attribution` 和 `regressed_unpredicted` 计算（后者的 `predicted_set` 也换成它）。

**语义变化（写进注释）**：`predicted_affected` 从「meta 对这个候选的专属声明」退化成「task 可救活性先验」。这是有意为之——Step 0b 证明 candidate-conditional 信号（diff 侧）为 0，meta 的专属声明是噪声，任务级先验才是真信号。与 `REPORT_visibility.md` §4.6「结构 > 数值」同一底层事实。

### 3.2 保留 meta 原始值（溯源）

- meta 的原始 `entry.predicted_affected` 仍写进 journal（不覆盖），新增一个 frontmatter `predicted_affected_meta` 或保留原名，用新名 `fliprate_predicted` 写替代值。二选一，倾向：**保留 `predicted_affected` = meta 原始，新增 `fliprate_predicted` = 翻转率 top-k**，归因改用 `fliprate_predicted`。这样 meta 仍能看到自己猜过什么，便于下一轮 meta 自我校正。

### 3.3 可选（建议，不阻塞主线）：gate 的 propensity 加权

`run.py` 里 gate 现在对 64 task 的 pass/fail 等权。加一个可开关（`--gate-task-weight fliprate|uniform`，默认 `uniform`，不动现有行为）：
- `fliprate` 模式下，高 flip_rate task 的「翻盘」权重低（预期内、信息量低），低 flip_rate task 的「翻盘」权重高（意外、信息量高）。
- 实现：把 `round_reward` 换成按 task 权重的加权均值，权重 `w = 1 − flip_rate`（或 logit 空间）。这影响 `score_and_gate` 的输入，不碰 gate 本身的 accept/reject 逻辑。

> 此条独立，先验证 3.1 有正效果再上，避免和 3.1 的归因改动混淆信号。

---

## 4. 离线验证（零 rollout，先做）

在写代码前，用现有日志离线验证 3.1 的实际增益：

1. 重算 `data/task_fliprate.json`（从 `data/kev_ranker_s0b/real_pairs.jsonl` 或 `cells`，口径 = B1）。
2. 对每个 run 的每个 R1-R5，构造两版 attribution：
   - **meta 版**：`compute_attribution(entry.predicted_affected, ...)`（现状）。
   - **fliprate 版**：`compute_attribution(fliprate_topk, ...)`（3.1）。
3. 指标：flip 预测的 precision/recall（对「本轮真的 flipped 的 task」），以及 attribution 里 `flipped` 的命中率。预期 fliprate 版显著更高（Step 0b 的 AUC +0.235/+0.249 已经指向）。
4. 产出 `data/predicted_affected_ablation/report.md`：两版 attribution 的命中率表 + 每个 stratum 的 k 敏感度（k ∈ {4, 8, 16, meta 原长}）。

**验收**：fliprate 版在 ALF 4B 与 WS 4B 的 flip 命中率都 ≥ meta 版（CI 不含 0）→ 才上 3.1 代码。

---

## 5. 交付物

```
data/task_fliprate.json                 # task_id → flip_rate（静态表，含 stratum）
data/predicted_affected_ablation/report.md   # §4 离线验证报告
recipe/agent_evolver/run.py              # 3.1 改动（+ 3.3 若做，加 --gate-task-weight）
```

## 6. 安全红线

- 不改 meta 的输出、不发任何新 API 调用、零 rollout。
- journal 若含 meta 调用请求头/key，加载时丢弃这些字段，不写进任何新文件。
- 所有新文件进 git 前跑 secret 扫描。
