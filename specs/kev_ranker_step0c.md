# SPEC：进化循环改成每轮 4 个分层候选，重放 3 次，三臂 3 seed

写给 coding agent。目的有两个。一个是用 3 个 seed 把"brief 里的结构和 kev 的数值各贡献多少"这个问题查清楚。另一个是让循环本身产生 kev ranker 需要的数据，也就是同一个 parent 下不同层的改动在同一批 task 上的结果。本 spec 不训练任何模型，kev 原样冻结使用。

## 0. 范围

要做的事情有六件。第一，先把 `SPEC_kev_ranker_s0b.md` 的第 2、3、4 节做完，也就是 R0 一致性检查、旧 run 重放、噪声底和 r_rep。第二，循环改成每轮 4 个候选，每个候选限定改一层，Gate 全部评估。第三，每个候选在 held-in 上评估 3 次，取均值。第四，失败类型换成有区分度的类型，ALFWorld 用 find、acquire、other，WebShop 用程序化的阶段标签。第五，跑三臂 3 seed，ALFWorld 4B 和 WebShop 4B。第六，把每轮所有候选的逐 task 结果导出成 kev 的训练数据格式。

不做的事情有三件。不训 kev。不跑 2B 和 0.8B。不改 agent 的 prompt 模板和解码设置，agent 侧和 `Visibility` 报告 §1.1 完全一致。

所有新行为都挂在新的命令行参数后面，参数不开时 `run.py` 的行为和现在逐字节一致，这是第 9 节的 G0。

## 1. 先做旧 run 的重放

按 `SPEC_kev_ranker_s0b.md` 第 2、3、4 节执行，产出 `r0_identity.json`、`cells.parquet`、噪声底和 r_rep。第 5、6 节不做。这一步的结果单独写一份 `replay_report.md`，里面要回答一个额外的问题。把 `data/predicted_affected_ablation/task_fliprate.json` 里翻转率最高的 10 个 task 拿出来，看它们在同一个 config 的三次重放里的不一致率，和其他 task 的不一致率并列报告。这个数字决定 §4.7 的翻转率归因到底在预测改动的效果还是在预测噪声。

## 2. 层的定义

harness config 的字段归到四个层，configuration、control、action、instruction，和 `decision_prior.py::AXIS_QUESTIONS` 里 lever 的四个选项一一对应。写 `recipe/agent_evolver/layers.py`，提供两个函数。

`layers_of(parent_config, candidate_config) -> dict[layer, int]`，返回每一层改动的字符数。

`project(parent_config, candidate_config, layer) -> config`，把候选里不属于这一层的改动全部丢掉，只保留这一层的改动，返回投影后的 config。投影是确定性的。

字段到层的对应表写在 `layers.md`，交给我看。0b 步已经写过一版 `layers_of`，在它的基础上改，不要重写。

## 3. 每轮 4 个分层候选

### 3.1 生成

每轮 meta 调用 4 次，每次的 brief 相同，只在末尾多一段约束，写明这一次只许改某一层，并列出这一层包含的字段。四次调用分别对应四个层，顺序固定。temperature 保持 1，四次调用相互独立。

meta 返回的候选先经过 `project` 投影到指定层。投影前后的 config 都记进 journal，投影丢掉了内容的要标 `violated=true` 并记下被丢掉的层。违反比例在报告里按层统计。

每个候选 meta 照常写 `predicted_affected`，四个候选各一份。

### 3.2 评估

四个候选都在 held-in 上评估，每个候选评估 `--eval-repeats` 次，默认 3，每次是 64 个 task 的完整 rollout，解码设置不变。候选的分数是 3 次均值，ALFWorld 用成功率，WebShop 用 mean reward，和现有 Gate 的度量保持一致。

Gate 的规则不改，只把输入换成 3 次均值。四个候选里分数最高的那个和历史最好比较，按现有的 tolerance 和 pass_count_noise 规则决定收或者不收。不收就保留 parent 进入下一轮。四个候选的全部结果不管收不收都记进 journal。

被收下的候选在 held-out 上也评估 3 次。这一步替代原来只对 held-in best 做一次 held-out 探针的做法，每轮都有 held-out 数字。

### 3.3 journal 格式

每轮一条记录，字段如下。

| 字段 | 内容 |
|---|---|
| `round` | 轮次 |
| `parent_hash` | 这一轮的 parent config |
| `candidates` | 长度为 4 的列表，每项见下表 |
| `accepted` | 被收下的候选在列表里的下标，没有收下填 null |
| `heldout` | 被收下的候选在 held-out 上 3 次的逐 task 结果 |

每个候选的字段。

| 字段 | 内容 |
|---|---|
| `layer` | 指定的层 |
| `raw_config` | 投影前 |
| `config` | 投影后 |
| `config_hash` | 投影后的 sha256 |
| `violated` | 是否有被丢掉的改动 |
| `layers_of` | 投影后每层改动字符数 |
| `predicted_affected` | meta 写的 |
| `repeats` | 长度为 3 的列表，每项是 64 个 task 的 pass、reward、步数、exit_reason 和轨迹文件路径 |
| `score` | 3 次均值 |

meta 的原始回复里的 rationale 存在单独的文件里，不放进 journal 的 `config` 字段，kev 的数据导出不读它。

## 4. 失败类型

### 4.1 ALFWorld

用 Mode A 的 find、acquire、other 三类。kev 的问题和 A1/A2 用的一样，`decision_prior.py` 里 Mode A 的实现还在就直接用，归档了就从 `archive/` 恢复。ground truth 从 `traj_data.json` 的 `plan.high_pddl` 反推，方法和 `a1a2_runner.py` 一致。每轮对所有失败轨迹同时记 kev 的答案和 ground truth，报告里给每轮的准确率，这是 A1/A2 在进化过程中的延续。

### 4.2 WebShop

程序化标签，从 WebShop 的 reward 分解推出来。WebShop 的 reward 由类型匹配、属性匹配、选项匹配、价格四部分组成，环境在购买时能给出各部分。标签按以下顺序判定，取第一个成立的。

| 标签 | 条件 |
|---|---|
| `no_search` | 没有执行过任何 search 动作 |
| `not_bought` | 搜过但在步数内没有 buy |
| `wrong_type` | 买了，类型匹配为 0 |
| `attr_mismatch` | 买了，类型对，属性匹配不满 |
| `option_mismatch` | 买了，类型和属性对，选项匹配不满 |
| `price_over` | 买了，前三项都对，价格超限 |

reward 等于 1 的不是失败，不打标签。reward 分解的取法写在 `webshop_labels.md`，如果环境接口拿不到分解，从 goal 的属性列表和购买时的商品页面自己算，算法写清楚。

kev 对 WebShop 失败轨迹回答一道六选一的题，选项就是上面六个标签。每轮记 kev 的答案和程序化标签，报告准确率。

### 4.3 区分度检查

两个环境的 baseline 失败轨迹上，任何一类的占比不能超过 70%，这是第 9 节的 G3。ALFWorld 用 A1/A2 的数字已知通过。WebShop 跑完 R0 之后先查这一条，不过就停下来向我报告，不要自己改标签定义。

### 4.4 brief 的结构

三个臂的 brief 结构沿用 `constant_prior_control.sh` 和 `decision_prior.py::constant_prior` 现有的实现，只把问题换成 4.1 和 4.2 的问题。lever 这道题保留，选项还是四个层。

## 5. 三臂 3 seed

| 臂 | brief 里的内容 |
|---|---|
| `llm` | 无 kev 结构，和现有 llm 臂一致 |
| `const` | 4.4 的结构，所有数值填 0.5 |
| `kev` | 4.4 的结构，数值是 kev 的答案 |

两个环境，ALFWorld 4B 和 WebShop 4B。seed 1234、2345、3456，seed 作用于 meta 的采样和 task 顺序，agent 仍是 temp 0。每个 run 6 轮。共 18 个 run。

三个臂的 R0 是同一个 config，R0 的评估只做一次，三个臂共用，重复次数照样是 3。

## 6. 算力预算

每轮的评估次数是 4 个候选乘 3 次加 held-out 3 次，共 15 次 64 个 task 的 rollout。先用现有 vLLM 部署量一次完整评估的 wall-clock，两个环境各量一次，按 18 个 run 乘 6 轮乘 15 次外推总时长，写进 `budget.md` 交给我，拿到我的确认再启动。

如果总时长超过预算，按这个顺序削减。先把 held-out 的重复次数从 3 降到 1。再把 held-in 的重复次数从 3 降到 2。候选数 4 不能减，seed 数 3 不能减。

## 7. 导出 kev 的数据

跑完之后写 `export_kev_data.py`，从 journal 导出 `kev_data/tuples.parquet`，每行是一个候选乘一个 task，字段如下。

| 字段 | 内容 |
|---|---|
| `env`, `arm`, `seed`, `round` | 来源 |
| `parent_hash`, `candidate_hash` | config |
| `layer` | 候选的层 |
| `diff` | 投影后的 config 改动，结构化，不含 rationale |
| `task_id` | task |
| `parent_traj` | parent 下这个 task 的轨迹文件路径，取 parent 被评估时的第一次 |
| `parent_pass_mean` | parent 下 3 次均值 |
| `cand_pass_mean` | 候选下 3 次均值 |
| `failure_type` | parent 下失败时的标签，4.1 或 4.2 的，通过的填 null |
| `kev_answer` | kev 对 parent 轨迹的答案分布 |

导出完成后直接在这份数据上重算 `SPEC_kev_ranker_s0b.md` 第 5 节的修复表和第 6 节的候选分差，不用再借 R0 到 R1 跨 run 的样本，每轮的 4 个候选就是候选组。结果写进 `kev_data/premises.md`。

## 8. 报告

`report.md` 分四部分。

第一部分是三臂比较。每个环境一张表，行是臂，列是 held-in 最好一轮的均值、held-out 对应的均值、3 个 seed 的标准差。另给 held-out 上 const 减 llm、kev 减 const 两个差值的 seed 级 bootstrap CI。两个差值是这个实验要回答的问题，写在报告最前面。

第二部分是层的统计。每个环境每个臂，四个层各被收下多少次，各层候选的平均分数，违反投影的比例。

第三部分是失败类型。每轮每个臂的类型分布，kev 对 ground truth 或程序化标签的准确率。

第四部分是第 7 节的前提重算。

所有 CI 用 seed 级 bootstrap，样本量在表里写明。单 seed 的数字一律不单独作为结论。

## 9. 检查

| 编号 | 内容 | 通过条件 |
|---|---|---|
| G0 | 参数不开时的回归 | 用 stub 模型跑 2 轮，journal 和改动前的代码逐字节相同 |
| G1 | 投影的单元测试 | 对每一层，构造跨层改动的候选，投影后 `layers_of` 只有这一层非零 |
| G2 | 一轮干跑 | 用 stub meta 和真实 agent 跑 1 轮，journal 字段齐全，15 次评估都有记录 |
| G3 | 类型区分度 | 两个环境 baseline 失败上最大类占比不超过 70% |
| G4 | 预算 | `budget.md` 经我确认 |

G0 到 G4 全部通过之后才启动 18 个 run。

## 10. 顺序

第 1 节的重放先跑，它不依赖任何代码改动，可以和第 2、3、4 节的开发并行。开发完成做 G0 到 G3。量预算做 G4。启动后先跑 ALFWorld 的 9 个 run，跑完先出第 8 节第一部分的 ALFWorld 表给我看，再跑 WebShop。

## 11. 运维和凭证

meta 的 API key 只从环境变量读，不写进任何脚本、配置、journal 或日志。meta 请求和回复落盘时去掉请求头。WebShop 服务地址在轨迹和日志里脱敏成占位符。报告、日志、commit 里不能出现 token、key、内部 URL 或机器主机名。所有新文件进 git 之前跑一遍 secret 扫描。
