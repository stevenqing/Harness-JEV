# FL L16 meta 进化结果（Harness-JEV Phase C）

日期：2026-10-04
方法：纯 harness 文本进化（system_prompt / guidance / history_window），模型权重永不训练。
agent = Qwen3.5-4B（vLLM :8200），decoding = spec §5（thinking off + max_tokens 4096 + temp 0.4/top_p 0.8/top_k 20）。
meta-agent = deepseek-v4-pro（mioffice anthropic SDK）。
难度背景见 [[gridgames-difficulty-map]]（`docs/gridgames-difficulty-map.md`）。

## held-in（evolve 256 关，1024 集/轮）6 轮曲线

| 轮 | pass | mean_reward | gate |
|---|---|---|---|
| R0 基线 | 53.9% | 0.5422 | — |
| R1 | 61.9% | 0.6240 | accept |
| R2 | 48.7% | 0.4890 | reject → revert R1 |
| R3 | 56.1% | 0.5662 | reject → revert R1 |
| R4 | 66.7% | 0.6756 | accept |
| R5 | **68.9%** | **0.6931** | accept ← best |

进化弧线 **+15.0pp / +0.151 reward**。2 次 regression（R2/R3）均被 gate 正确拦截并回滚到 best。无 crash。

## held-out 泛化（test 128 关，unseen，用 R5 config）

| 集 | pass | mean_reward |
|---|---|---|
| held-out R5 | **65.0%** (333/512) | 0.6579 |

- held-in R5 68.9% → held-out R5 65.0%，gap = **3.9pp / 0.035** → **无过拟合**，进化成果在 unseen 上基本保留。
- 相对 held-in 基线 53.9% → held-out 65.0%，即便在没见过的关卡上也净 +11pp（注：未跑 held-out 基线，绝对增益是按 held-in 基线推断；若需更严可补跑 `--base-config` 无、`--split test` 的 R0）。

## 赢家 harness（R5）——meta-agent 独立收敛出的三招

在 R1「5 步安全清单」基础上，R5 又叠了三个对应真实 FL 失败模式的杠杆：

1. **列坐标双向核对**（step 1 & 5）：从左往右 + 从右往倒数，两边一致才定列坐标 —— 治「读盘数错列」。
2. **防死循环逃逸**（step 4 末尾）：看 "Last moves"，来回弹同一对格子 = 陷口袋 → 先走一个暂时增加距离的 SAFE 步逃出，短绕路好过绕到耗尽步数 —— 治 FL 典型 pocket 打转。
3. **只看 P 的 4 邻居 + 强制 `<action>` 收尾**：不转录全盘、只写 4 个邻格的 (row,col)+符号并标 SAFE/HOLE，选缩 Manhattan 距离的方向 —— 压缩「reason step-by-step」的全盘扫描。

## 结论

FL L16 是理想中间 tier（基线 ~54%），纯 harness 进化把它拉到 ~69%，且泛化 gap 仅 3.9pp。核心印证：**瓶颈是 prompt 措辞（结构/核验/破环），不是 decoding**；meta-agent 无需外部提示即在 harness 文本层面独立发现了这些杠杆。

## 下一步

SK L16 同样的 6 轮 meta evolve（注意：SK L16 基线仅 21.1%、invalid 64.7%，比 FL L16 硬得多——Sokoban 大棋盘 + 冗长推理更易截断，是对「压缩推理」杠杆的更强考验）。
