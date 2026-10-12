# 进度诊断：kev 卡位 vs oracle 里程碑 对齐 + G3 验收

进度诊断 SPEC 的验证半边（§4.2 kev judge + §4.4 validation）。核心问题一句话：**kev 从轨迹文本判「agent 走到哪一步了」，能不能恢复 oracle 给的真实卡位？** 恢复得了 = kev 是可靠的进度法官（方法有效）；恢复不了 = 又退化成「无脑答一个位置」，停（不做重放、不做 24 跑）。

## 0. 两关结构（meta 在哪）

把「meta 分解」和「kev 判进度」拆开，它们验证的确定性不同：

| 关 | 干什么 | 需不需要 meta | 可验证性 |
|---|---|---|---|
| **第一关** | 固定条件（oracle 派生）+ kev 判 → 卡位 k vs oracle 卡位 m | **不需要** | 完全可验证 |
| **第二关** | meta 读 goal 自由分解（更细 / 迁无 oracle 环境）+ kev 判 | 需要 | 只能验粗粒度对齐，细粒度不可验 |

**第一关是本文档的全部内容。** 第一关过了（kev 是可靠进度法官），第二关的「meta 自由分解」才值得谈——因为只有先证明 kev 能判进度，meta 分解出来的更细条件才有被 kev 正确执行的底座。

> 为什么第一关不需要 meta：oracle 已经把 meta 要拆的东西全给了。
> - ALFWorld：`alf_requirement` 给 target（"mug"）+ 要求动作（put/cool/…）。
> - WebShop：goal dict 给 type/attributes/options/price。
> meta 分解在这个 testbed 的产出 = oracle 已知的东西，还多一层「拆错」的不确定性。

## 1. 固定条件（oracle 派生，第一关）

条件**从 oracle 派生**，具体值从 oracle 的目标字段填，不经过 meta：

| benchmark | 里程碑模板（M 条） | 条件措辞的具体值来源 | oracle 来源 | m 取值 |
|---|---|---|---|---|
| ALFWorld | ① 找到目标物体 ② 完成目标任务动作 | `alf_requirement` 的 target / req_verb+req_obj | `failure_labels.alfworld_label` → `find`/`acquire` | 1..2 |
| WebShop | ① 搜到商品 ② 点进类型匹配商品 ③ 属性匹配 ④ 选项匹配 ⑤ 价格不超限 ⑥ 完成购买 | goal dict 的 type/attributes/options/price_upper | `webshop_labels.ws_label` → 六路 | 1..6 |

条件写成 kev 判得动的状态句，例：

- ALF ①「agent 拿到了 {target}」（`take {target}` 语义，非字面 parse）；②「agent 完成了 {req_verb} {req_obj}」。
- WS ③「选中的颜色是 {goal color}、尺码是 {goal size}」；⑤「价格 ≤ {price_upper}」。

### 1.1 oracle 标签 → 卡位序号 m

- **ALFWorld**：`find` → m=1（没 `take <target>`）；`acquire` → m=2（`take` 了但没做完要求动作）；`other`（无 pickup / 拿到即完 / 已完成要求）**从验证集剔除**。
- **WebShop**：见 `webshop_oracle.md` §2 的重排表（`no_search`→1 … `not_bought`→6）。

## 2. 卡位定义

- **oracle 卡位 m**：每个失败轨迹的第一个未达成里程碑序号（ALF 1..2，WS 1..6）。
- **kev 卡位 k**：kev 逐里程碑判 `p(满足)`（`KevBackend.decide`，每 milestone 一条 `JudgeQ` noul）。`k = min{ i : p_i < 0.5 }`；若全 ≥ 0.5 则 k = M+1（「看起来成功了」，但轨迹实际失败，记作错配）。

固定条件让 k 与 m 落在**同一序号空间**，逐序号直接可比，无映射歧义。

## 3. 对齐指标（§4.4 验证核心）

在全部「可覆盖」的失败轨迹上算（ALF 剔除 `other`；WS 全六路可覆盖）：

| 指标 | 定义 | 作用 |
|---|---|---|
| **覆盖率** | 有 oracle 标签的失败轨迹 / 全部失败轨迹 | 门槛：ALF 剔除 `other` 后应 ≥ 90%，否则 plan 质量太差 |
| **卡位准确率** | P(k == m) | 方法有效性的**主指标** |
| **分布 gate** | kev 卡位分布是否单一化（见 §4） | 退化解检测 |

准确率的解读锚点（和已知退化的对照）：

- 之前 lever 行为问题：18/25 题退化（≥70% 塞进单选项），lens 更是 99.7% 「mechanical」。kev 卡位**必须**显著脱离这个地板。
- 卡位准确率显著高于「常数基线」（always 猜最常见的 m）才说明 kev 读了轨迹、不是背 base rate。
- **GO**：准确率明显过常数基线 + 分布贴合 oracle（JS 小）→ kev 是可靠进度法官，进 §3 四候选 / §5 24 跑（再谈第二关 meta 分解）。
- **NO-GO**：准确率≈常数基线（背 base rate）或 kev 分布塌到错误位置（JS 大）→ 和 lever 一样的死法，停，不烧重放/24 跑预算。

## 4. 分布 gate：准确率 vs 基线 + JS(kev‖oracle)（两 benchmark 统一）

之前 lever 退化 =「kev 无脑答 mechanical」。进度诊断对应的退化有两种死法，用两个信号分别抓：

| 退化死法 | 检测信号 | 阈值 |
|---|---|---|
| 背 base rate（无脑猜最常见的 m） | **准确率 vs 常数基线** | 准确率须显著 > 常数基线 |
| 塌到错误位置（无脑猜某个 ≠ oracle 的位置） | **JS(kev‖oracle)** | < 0.2 |

**不拍「单 milestone ≤ 70%」的扁平上限**——oracle 本身就可能 >70% 集中（ALF `find` 实测 71.4%，WS `not_bought` 可能大头），正确 kev 必然跟着集中。扁平上限会误伤完美 kev。判断退化只看「kev 是否贴着 oracle」：kev 若「无脑 milestone 1」而 oracle 集中在别的位，JS 散度立刻爆掉；kev 若「无脑猜最常见 m」，准确率就退到常数基线。两个信号都不依赖 oracle 分布是否均衡。

## 5. 落地顺序（第一关，§4 链）

1. **建条件**：oracle 派生固定条件（`alf_requirement` / goal dict 填具体值）→ `decomp/<env>.jsonl`（每 task M 条状态句）。**无 meta 调用。**
2. **kev judge**：`KevBackend.decide`，每失败轨迹逐 milestone 一条 noul `JudgeQ` → 卡位 k。
3. **oracle**：复用 `failure_labels`（ALF）/ `webshop_labels`（WS）→ 卡位 m。
4. **validation**：覆盖率 + 卡位准确率 + 分布 gate → G3 出 GO/NO-GO。

安全红线（SPEC §11）：条件/回复落盘去请求头；WebShop 服务地址在轨迹/日志脱敏成占位符；token 不进任何产物。
