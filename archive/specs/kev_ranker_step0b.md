# SPEC：kev ranker 第 0b 步，三个前提的检验

> 写给 coding agent。第 0 步的结论是 NO-GO，原因是验证设计泄漏，不是方法本身的结论。这一步**不训练任何模型**。它只回答三个问题：(1) 交互项是否高于重放噪声；(2) 交互是否能从失败类型和改动的层读出来；(3) 候选之间的分差是否大到值得排序。三个问题都用现有 run 的数据加一轮重放来回答。第 8 节的判定出来之后，才决定要不要写训练的 spec。

## 0. 范围

要做的事情五件：①确认各臂 R0 的 config 是否逐字节相同；②把每个 run 每一轮的 config 在 held-in 上重放两次，得到每个格子三次结果；③用重放算噪声底和交互项的复现相关；④列失败类型×改动层的修复表，并用模型阶梯检验它的预测力；⑤在 R0→R1 的跨 run 候选上算候选分差和随机挑选的 regret，并量评估和 meta 调用的耗时。

不做的事情四件：不训练 kev 的 head；不调用 meta、不发任何新的 API 请求；不改进化循环的代码；不碰第 0 步的交付物，新结果写到新目录。

## 1. 数据来源

现有 run 共 13 个，覆盖 ALFWorld 和 WebShop 的 4B 和 2B，臂有 llm、kev 软先验、kev enforce 和 Mode A。每个 run 6 轮，每轮一个 config，每个 config 在 64 个 held-in task 上跑过一次。journal 里有每轮的 config、逐 task 结果、meta 写的 `predicted_affected`、以及 Gate 当时的历史最好 config。先验臂的 run 里还有 kev 对每条失败轨迹的三道题回答。

加载代码**沿用第 0 步** `Harness-JEV/data/kev_ranker_s0/` 里的 loader，不要重写。本步所有产出写到 `Harness-JEV/data/kev_ranker_s0b/`。

一个环境加一个模型规模叫一个 **stratum**，共四个：ALFWorld 4B、ALFWorld 2B、WebShop 4B、WebShop 2B。所有统计量按 stratum 分别算，不跨 stratum 合并。

## 2. R0 的一致性检查

同一 stratum 下各 run 的 R0 config 应是同一个裸 harness。序列化后算 sha256，写进 `r0_identity.json`：stratum、每个 run 的 hash、是否全部相同。若不同，输出逐字段 diff，并**停在这里向我报告**。这一步决定 §4 噪声估计能不能用，必须先做。

同时检查 agent 侧解码设置在各 run 是否一致：温度、max_tokens、采样参数、vLLM 版本、batch 设置，写进同一文件。

## 3. 重放

对每个 run 的每一轮 config，在 held-in 64 task 上各重放两次，与原一次合成每个格子三次。解码设置与原 run 完全相同，temp 0。同一台机器、同一 vLLM 版本、batch 设置与原 run 一致。约 78 config × 64 task × 2 次。

每次重放记录逐 task 的 pass、reward、步数、exit_reason、完整轨迹，格式与原 run 一致。轨迹要存（§5 用）。

输出 `replays/<run>/<round>/rep{1,2}.jsonl`，汇总成 `cells.parquet`，每行一个格子：

| 字段 | 内容 |
|---|---|
| `stratum` | 环境加规模 |
| `run` | run 标识 |
| `round` | 轮次 |
| `config_hash` | config sha256 |
| `task_id` | held-in task |
| `pass_0` `pass_1` `pass_2` | 原始一次 + 两次重放 |
| `reward_0` `reward_1` `reward_2` | WebShop 用 reward，ALFWorld 填 pass |
| `pass_mean` | 三次均值（软标签） |

重放中某次 rollout 报错 → 重跑这一个 task，不整轮重跑。报错次数与原因写进 `replay_log.md`。

## 4. 前提一：交互项要高于重放噪声

### 4.1 噪声底
每 stratum 三个数：①格子级不一致率（三次不全相同的格子占比）；②config 级 pass count 标准差（每 config 三次通过数标准差，取所有 config 均值，与 Gate 默认 `pass_count_noise_threshold=3` 并排报告）；③WebShop 另算 mean reward 标准差。

同 stratum 各 run 的 R0 是同一 config，把所有 R0 重放合并当同一 config 的多次重放单独报告，与 §2 hash 结果一起看。

### 4.2 复现相关
每 stratum 把每格子三次拆两组拟合/复现：取重放 A，在 A 上用 task 均值 + config 均值拟合加性模型得残差 e_A；取重放 B 同理得 e_B；算 e_A 与 e_B 的 Pearson 相关。三对两两配对取均值。CI 用按 run 的 cluster bootstrap（1000 次）。

这个相关系数 r_rep 是该 stratum 上任何模型能达到的交互项预测上限，后续所有模型都对着它报告。

另用三次均值做双因素方差分解，报告 task 主效应、config 主效应、残差三块方差占比，残差里按 4.1 噪声底估计可归噪声的部分。

### 4.3 判定
r_rep 的 CI 下界 > 0 记为通过。**ALFWorld 4B 和 WebShop 4B 都不通过 → 整条线停，写报告。**

## 5. 前提二：交互要能从失败类型和改动层读出来

### 5.1 真实 parent→candidate 对
从 journal 重建每轮真实对：第 r 轮 candidate = 本轮评估的 config，parent = Gate 本轮之前的历史最好 config。每 stratum 约 15 对，共约 60 对。写 `real_pairs.jsonl`（stratum/run/round/parent_hash/candidate_hash/predicted_affected）。

对每对，parent 三次重放 `pass_mean<0.5` 的 task 记失败集，`>0.5` 记通过集，`==0.5` 丢弃并计数。失败集标签 = candidate 的 `pass_mean`（翻转问题）；通过集标签 = 1 − candidate `pass_mean`（掉落问题）。本节**只用翻转问题做判定**，掉落问题只报告。

### 5.2 失败类型
用 kev 现有三道题回答。先验臂 run 的 journal 已有；其他臂 run 对每个 parent config 的每条失败轨迹用**冻结 kev 离线跑一遍三道题**，得 `lens_failure`、`lens_capability_gap`、`lever` 分布。kev 权重用当前部署那份，记录 hash。轨迹用三次重放里的原始那一次。

每条失败轨迹得一个类型分布 + 一个 argmax 类型。

### 5.3 改动层
candidate 相对 parent 的 diff 按 config 字段归四个层：configuration、control、action、instruction（对应 lever 四选项）。写确定性函数 `layers_of(diff) -> dict[layer, size]`（size=该层改动字符数），主层=size 最大层。多层同改的对要计数并报告。对应规则写进 `layers.md`（**我要看**）。

### 5.4 修复表
每 stratum，把真实对失败集样本列成表：行=失败类型 argmax，列=主层，格=翻转率 + 样本数。用似然比检验比较两个 logistic 模型（行效应+列效应 vs 再加行×列交互），报 p 值。样本按 task×run 去重后的单元数一起报。

### 5.5 模型阶梯
三个模型，都在翻转问题上，按 run 轮流留出评估（13 run 全轮一遍）：
- **L1 加性**：task 侧 = 该 task 在训练 run 的历史翻转率（无历史填 stratum 均值）；diff 侧 = 四层改动字符数 + 改动字段数 + 新增规则条数。逻辑回归。
- **L2 加性+修复表**：L1 特征 + 失败类型分布 × 主层外积（lens_failure 选项数 × 4 特征）。逻辑回归，L2 正则，强度训练 run 内 CV 选。
- **L3 本步不做**（留训练 spec）。

两个 baseline：task 历史翻转率单独；`predicted_affected` 单独（仅 journal 有它的对上算）。

### 5.6 评估
每模型报：留出 run 的 log loss；固定 candidate 对失败 task 排序的 AUC（按对算再平均）；按 run 的 cluster bootstrap CI。

### 5.7 判定
两条同时满足记通过：①5.4 交互项似然比检验在 4B 两个 stratum 至少一个 p<0.05；②5.5 L2 留出 run AUC 高于 L1（差值 CI 不含 0）且高于 `predicted_affected`。不通过 = 失败类型×改动层假设无预测力，整条线停。

## 6. 前提三：排序要能改变决策

### 6.1 候选分差
同 stratum 各 run R0 相同、R1 是同一 parent 出发的不同候选。对每 stratum，用 R1 config 三次重放均值算 held-in 分数（ALFWorld 成功率，WebShop mean reward）。报候选数、最高−最低差、随机挑一个相对最高的期望损失，与 4.1 config 级噪声标准差比。

### 6.2 耗时
重放日志取每 stratum 一次完整 held-in 评估 wall-clock；journal 时间戳取 meta 写一个候选的耗时（无时间戳写缺失，不发新 meta 请求）。并列报告。

### 6.3 判定
候选分差 > 两倍噪声标准差的 4B stratum 至少一个 → 排序用途通过。不通过 = 候选差别在噪声内，kev 排序挑不出差别，排序用途停；但 kev 替 meta 写 `predicted_affected` 的用途不受影响，由 §5 结果决定。评估耗时 < meta 耗时的 stratum 单独标出（全部评估比排序便宜，排序只在放大任务池后才有意义，写进报告）。

## 7. 统计规范

样本计数报两个数：配对数 + task×run 去重单元数。所有 CI 用按 run cluster bootstrap（1000 次，95%）。随机操作固定种子写进报告。评估只用 5.1 真实对，不用跨多轮 Ri→Rj 对。

## 8. 判定和交付

`report.md` 第一节三行判定（前提一/二/三各一行：通过或不通过 + 依据数字），之后按 §4/5/6 顺序给全部表格和图。

| 前提 | 不通过时 |
|---|---|
| 一 | 整条线停，报告噪声底和 r_rep |
| 二 | 整条线停，报告修复表和模型阶梯 |
| 三 | 排序用途停，诊断用途按前提二结果决定 |

三个都通过才写训练 spec。交付目录：

```
data/kev_ranker_s0b/
  r0_identity.json
  replays/
  cells.parquet
  replay_log.md
  real_pairs.jsonl
  failure_types.parquet
  layers.md
  repair_table/
  ladder/
  decision_value/
  report.md
```

## 9. 顺序

§2 先做，不通过就停。§3 重放最长，先跑 4B 两个 stratum，跑完做 §4，4B 不通过就不跑 2B。4B 通过再跑 2B 重放，然后做 §5、§6。§5.2 kev 离线推理可与重放并行。

## 10. 运维和凭证

重放和 kev 推理都在本地 GPU，不需任何 API key。journal 若存了 meta 调用请求头/key，加载时丢弃这些字段，不写进任何新文件。报告/日志/commit 不出现 token、key、内部 URL、机器主机名。重放轨迹若含 WebShop 服务地址，脱敏成占位符再存。所有新文件进 git 前跑一遍 secret 扫描。
