# §4 离线验证 — task 历史翻转率 vs meta `predicted_affected`

seed=20261011, cluster bootstrap by run, B=1000

口径：pass/fail = Step-0b B1（ALF 三复本均值、WS 三复本奖励均值，阈值 0.5）；flip = parent 失败 → candidate 通过。只评估 parent 失败的 task。


## 1. 验收结论

**GO**

- ALFWorld 4B：fliprate vs meta AUC 差值 CI95 下限 > 0（通过）

- WebShop 4B：fliprate vs meta AUC 差值 CI95 下限 > 0（通过）


（验收标准：fliprate 版在 ALFWorld 4B 与 WebShop 4B 的 AUC 都 ≥ meta 版，且差值 CI 下限 > 0。）


## 2. AUC 对比（主指标）

| stratum | meta AUC | meta CI95 | fliprate AUC | fliprate CI95 | diff (fr−meta) | diff CI95 | n_pairs(成对) |
|---|---|---|---|---|---|---|---|
| alfworld_4b | 0.508 | (0.477, 0.538) | 0.784 | (0.756, 0.813) | 0.273 | (0.226, 0.322) | 51 |
| alfworld_2b | 0.566 | (0.446, 0.642) | 0.950 | (0.914, 0.966) | 0.387 | (0.324, 0.455) | 10 |
| webshop_4b | 0.558 | (0.505, 0.602) | 0.932 | (0.918, 0.943) | 0.368 | (0.334, 0.429) | 14 |
| webshop_2b | 0.519 | (0.411, 0.661) | 0.967 | (0.949, 0.987) | 0.445 | (0.323, 0.525) | 8 |


## 3. precision@k / recall@k（k = 每对 |predicted_affected ∩ parent失败|）

| stratum | meta P@k | fliprate P@k | P@k diff | meta R@k | fliprate R@k | R@k diff |
|---|---|---|---|---|---|---|
| alfworld_4b | 0.389 | 0.574 | 0.186 | 0.481 | 0.705 | 0.224 |
| alfworld_2b | 0.076 | 0.328 | 0.252 | 0.290 | 0.823 | 0.533 |
| webshop_4b | 0.236 | 0.598 | 0.362 | 0.319 | 0.592 | 0.273 |
| webshop_2b | 0.252 | 0.406 | 0.155 | 0.296 | 0.742 | 0.446 |
（P@k/R@k 为跨对均值；diff 为 fliprate − meta，成对，只在两版都有定义的 pair 上算。）


## 4. k 敏感度（fliprate 版 top-k 的 precision/recall，固定 k）

| stratum | k | P@k | R@k | n_pairs |
|---|---|---|---|---|
| alfworld_4b | 4 | 0.725 | 0.380 | 59 |
| alfworld_4b | 8 | 0.587 | 0.581 | 56 |
| alfworld_4b | 16 | 0.480 | 0.810 | 49 |
| alfworld_2b | 4 | 0.333 | 0.619 | 15 |
| alfworld_2b | 8 | 0.242 | 0.774 | 15 |
| alfworld_2b | 16 | 0.163 | 1.000 | 15 |
| webshop_4b | 4 | 0.575 | 0.638 | 20 |
| webshop_4b | 8 | 0.419 | 0.868 | 20 |
| webshop_4b | 16 | 0.250 | 0.995 | 20 |
| webshop_2b | 4 | 0.583 | 0.779 | 15 |
| webshop_2b | 8 | 0.383 | 0.958 | 15 |
| webshop_2b | 16 | 0.200 | 1.000 | 15 |


## 5. 样本量

| stratum | n_runs | n_pairs | meta AUC 可用对 | fliprate AUC 可用对 | 成对 diff 可用对 |
|---|---|---|---|---|---|
| alfworld_4b | 13 | 59 | 51 | 58 | 51 |
| alfworld_2b | 3 | 15 | 10 | 11 | 10 |
| webshop_4b | 4 | 20 | 14 | 19 | 14 |
| webshop_2b | 3 | 15 | 8 | 14 | 8 |


## 6. 口径与注意事项

- flip_rate 为**全样本**（所有纳入 run 的相邻 parent→candidate 对）计算，含自包含，因此 fliprate AUC 是 in-sample 上界（spec §4.1 明确口径）。
- leave-one-run-out（LOO）翻转率 AUC 作为泄漏/B1 对齐检查：
  - alfworld_4b: LOO AUC = 0.745（n=13 runs）
  - alfworld_2b: LOO AUC = 0.798（n=3 runs）
  - webshop_4b: LOO AUC = 0.810（n=4 runs）
  - webshop_2b: LOO AUC = 0.805（n=3 runs）
  - 4B 两 stratum 的 LOO AUC 应对齐 Step-0b B1（0.747 / 0.810）；全样本值高于 LOO 即自包含增益。
- parent 通过（pass_mean ≥ 0.5）的 task 不进评估（回归问题，spec §4 只管 flip）。
- pass_mean 恰等于 0.5 的 cell 丢弃（与 B1 一致）。


