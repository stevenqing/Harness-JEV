# Gridgames 难度地图（FrozenLake / Sokoban）

日期：2026-10-03
方法：均分采样 32 关 × 4 集 = 128 集/格，Qwen3.5-4B（vLLM :8200），
decoding = spec §5（thinking off + max_tokens 4096 + temp 0.4/top_p 0.8/top_k 20），
baseline prompt = 统一 ALFWorld ReAct 协议（"reason step-by-step" + `<think>`/`<action>`），无 gridgames 特调。

## 全表

| 格子 | pass | mean_reward | invalid(不可解析步占比) |
|---|---|---|---|
| FL L4 | 84.4% (108/128) | 0.8594 | 3.6% |
| FL L8 | 60.2% (77/128) | 0.6062 | 6.9% |
| FL L16 | **51.6%** (66/128) | 0.5170 | 14.6% |
| FL L32 | 10.2% (13/128) | 0.1016 | 45.3% |
| SK L4 | 85.2% (109/128) | 0.8646 | 25.9% |
| SK L8 | 67.2% (86/128) | 0.7264 | 39.2% |
| SK L16 | 21.1% (27/128) | 0.2907 | 64.7% |
| SK L32 | 3.1% (4/128) | 0.0707 | 80.7% |

split 容量：每 (game, tier) 下 evolve(held-in)=256、test(held-out)=128、gate=64。

## 结论

1. **原「91.7%」是 L4 太简单 + 简单偏向采样的虚高**。均分采样下 FL L4 = 84.4%。L4 对两游戏都是饱和区，无进化空间。
2. **两条单调难度梯度成立**，且 invalid 率随 tier 单调上升：
   - FL：84.4 → 60.2 → 51.6 → 10.2，invalid 3.6 → 6.9 → 14.6 → 45.3。
   - SK：85.2 → 67.2 → 21.1 → 3.1，invalid 25.9 → 39.2 → 64.7 → 80.7。
3. **invalid 率曲线 = 「compress reasoning / 别扫全盘」杠杆的量化画像**。"reason step-by-step" 诱导全盘扫描 → 在 `<action>` 前耗尽 max_tokens（finish_reason=length）→ parse=None。
4. **进化 tier 首选 FL L16**（51.6%，invalid 14.6%）：卡在 50% 附近、成功/失败各半有信号；invalid 只有 14.6%，失败大头是真走错（可被更好的推理救回）而非截断机械失败。FL L8（60.2%，invalid 6.9%）是次选。
5. **SK 不进 L8+**：同 tier invalid 是 FL 的 3–4 倍（棋盘多行渲染 → 更长思考 → 截断），失败大头是机械失败，meta 改 prompt 会被 parse 失败淹没。

## 后续

- FL L16 6 轮 meta 进化（held-in=evolve 256 关）→ held-out=test 128 关评测。
- 头部空间是 prompt 措辞本身（不是 decoding），正是 meta-agent 该发现的杠杆。
