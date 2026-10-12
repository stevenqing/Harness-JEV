# SPEC：ScienceWorld 接入 agent_evolver

写给 coding agent。目标是把 ScienceWorld 接进 `recipe/agent_evolver`，让它和 ALFWorld、WebShop 用同一套 runner、同一套解码、同一种 prompt 形状、同一种轨迹格式。本 spec 只做接入和 baseline，不跑进化，不训练任何模型。第 9 节的检查全部通过、baseline 报告交给我看过之后，才进入进化实验。TextCraft 的接入在另一份 spec 里，两份可以并行做，但共用的部分（第 8 节的 `render_for_reader` 和 runner 注册方式）要保持一致。

## 0. 范围

要做的事情有六件。第一，把环境接进来，用一个固定大小的进程池管理 JVM。第二，选定任务类型，固定 held-in 和 held-out。第三，写 prompt 模板，结构和 `agents.py:ALFWORLD_TEMPLATE` 一致。第四，写动作解析，并处理歧义菜单。第五，写一个从环境结构推出失败标签的模块。第六，用 Qwen3.5 的 4B、2B、0.8B 各跑一次 R0 baseline。

不做的事情有三件。不跑 meta 进化。不改 kev。不碰 ALFWorld、WebShop、gridgames 现有的任何代码路径。

## 1. 环境来源和安装

环境用 PyPI 上的 `scienceworld==1.2.3`，版本必须固定。git 主线上有一个还没发布的 1.3.0 改动，会让对象顺序变成确定性的 UUID 顺序，进而改变 gold 路径、歧义菜单和 observation 的顺序。需要 Java 11 或更高版本，OpenJDK 17 或 21 都可以。

包装代码以腾讯 RLVMR 为起点，仓库 https://github.com/Tencent/DigitalHuman/tree/main/RLVMR 。它是 verl-agent 的 fork，已经有 ScienceWorld 的环境包，prompt 形状也和我们一样。参考这几个文件：`code/agent_system/environments/env_package/sciworld/envs.py`、`code/agent_system/environments/env_package/sciworld/projection.py`、`code/agent_system/environments/prompts/sciworld.py`，以及 `code/agent_system/environments/env_manager.py` 里的 `SciWorldEnvironmentManager`。用得上的部分放进 `recipe/agent_evolver/envs/sciworld/vendor/`，保留原 LICENSE，并在 `VENDOR.md` 里写明来源 URL、commit 和改动列表。

RLVMR 的整数 task 编号对应的是 1.2.x 里按字母排序的任务顺序，不是论文里的顺序。我们不用整数编号，一律用任务名。

## 2. 环境设置

### 2.1 simplification

用这四个标签，在字符串里逐个写出来：`teleportAction,openDoors,selfWateringFlowerPots,noElectricalAction`。这四个就是 `easy` 预设的内容。不要写成 `easy`，也不要把 `easy` 和其他标签拼在一起写。Python 端能接受这种写法，但 Scala 端会返回 `ERROR: Unknown simplification label: easy`，并给出 -100 分。

### 2.2 步数

agent 的步数上限 `max_steps` 设为 40。环境自己的 `envStepLimit` 设为 100。后者一定要大于前者，否则环境会因为自身的步数上限提前把 `done` 置为 true，和任务完成混在一起。

### 2.3 进程池

每个 `ScienceWorldEnv` 都会通过 py4j 起一个 JVM，启动大约 2 秒，占用 250 到 290 MB 内存。所以不要每局新建一个环境。做法是开一个固定大小的进程池，默认 16 个 worker，每个 worker 持有一个环境，每局开始时用 `env.load(task_name, variation, simplification)` 重新加载。worker 用 spawn 方式启动。

`env.step()` 每一步都会重新计算完整的合法动作列表、look 和 inventory，按任务不同每步耗时 34 到 218 毫秒。baseline 阶段直接用 `env.step()`，先在报告里记下每步耗时。如果一局跑完的时间超过 30 秒，再考虑改成精简的 step（直接调 server 的 step、getScore 和 getCompleted），那是一个单独的优化，要先做 A/B 证明结果完全一样。

## 3. 任务选择和切分

### 3.1 选哪些任务类型

ScienceWorld 一共 30 类任务。有些任务的 gold 路径比我们的步数预算还长，小模型不可能完成，留着只会给出一堆 0 分。写 `envs/sciworld/select_tasks.py`，做这几步：在 2.1 的 simplification 下，对每类任务取前 5 个 train 变体，生成 gold 动作序列（`env.load(..., generateGoldPath=True)` 再调 `get_gold_action_sequence()`），记下每个序列的长度，以及按这个序列重放以后的最终分数。

一类任务被选入，要同时满足两个条件：gold 序列长度的中位数不超过 25；5 个 gold 序列重放后都能拿到 100 分。

输出 `envs/sciworld/task_selection.json`，列出每一类任务的 gold 长度中位数、重放分数和是否选入。按以往的测量，`inclined-plane-friction-unnamed` 和两个 mendelian 任务的 gold 路径超过 100 步，大概率会被排除。

### 3.2 切分

held-in 从选入任务的 train 变体里取，held-out 从 test 变体里取，各 64 局。每类任务分到的局数尽量相等。整除不尽的余数，分给变体最多的那几类。每类任务取变体时，按变体编号从小到大取前 k 个，不要随机抽。结果写进 `envs/sciworld/splits.json`，每一项是 `{episode_id, task_name, variation, split}`。

## 4. 环境包装

写 `envs/sciworld/env.py`，接口和现有的包装一致，即 `reset(episode_id) -> (obs, info)` 和 `step(action_text) -> (obs, reward, done, info)`。

### 4.1 observation

reset 返回的 observation 是任务描述加上初始 look。之后每一步的 observation 就是环境返回的原始文本。

### 4.2 可用动作块

完整的合法动作组合每个状态有几百到几千条，厨房里一个状态大约就有 7k token，不能像 ALFWorld 那样整个放进 prompt。baseline 只给两样东西。一样是动作模板，来自 `get_possible_actions()`，去掉 `reset task`。另一样是当前能看到的物体，来自 `get_possible_objects()`。

完整的合法动作列表仍然每一步都要拿到（`info['valid']`），只用来记录 `in_valid`，也就是模型这一步的动作是否在合法列表里，不展示给模型。

### 4.3 歧义菜单

有些动作会触发一个编号菜单，例如 `Ambiguous request: Please enter the number...`，下一步必须输入一个数字。baseline 里由包装层自动回答 `0`，这一步不计入 agent 的步数，但要记进 info，字段名 `auto_disambiguated`。是否改成把菜单交给模型自己选，是一个 harness 变量，配置名 `ambiguity_mode`，可选 `auto0` 或 `show`，baseline 用 `auto0`。

### 4.4 分数和结束

环境每步返回一个 0 到 100 的分数，由子目标累计而来。focus 到错误的物体时，分数直接变成 -100，并且 done 为 true，这一步不可逆。

每局结束时记录这几个数：`final_score_raw` 是最后一步的原始分数；`score` 是 `max(final_score_raw, 0) / 100`；`max_score` 是整局出现过的最高分；`success` 为 true 当且仅当 `getCompleted()` 为 true 并且分数是 100。

结束条件有三个，满足任一就结束：环境返回 done；完成任务；达到 `max_steps`。

## 5. prompt 模板

在 `agents.py` 里新增 `SCIWORLD_TEMPLATE`，结构对齐 `ALFWORLD_TEMPLATE`。system 用 Qwen3.5 默认 system。

```
You are an expert agent operating in the ScienceWorld environment, a text-based simulation of science experiments. Your task is to: {task_description}
{skills}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Valid action templates (replace OBJ and LOC with objects or locations): {action_templates}
Objects you can currently see: {visible_objects}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose one valid action for current step and present it within <action> </action> tags.
```

`{skills}` 是 guidance 槽，baseline 为空，进化时由 meta 填。`history_length` 默认为 2。历史里只放动作和 observation，不放 think。

baseline 不加任何关于 focus 的警告。错误的 focus 会直接让这一局得 -100，这本身就是一个 harness 可以去修的失败，留给进化。

## 6. 动作解析

写 `envs/sciworld/projection.py`。从输出里取最后一对 `<action>...</action>` 之间的文本，去掉首尾空白，把连续空格合并成一个。不做大小写转换，也不做任何模糊匹配，原文交给环境。

没有 `<action>` 标签或标签里为空，记为解析失败。这一步不调用环境，返回固定的 observation `Invalid action format.`，并消耗一步。

RLVMR 的 projection 还会在输出含中日韩字符时判为无效。我们不加这条规则。

## 7. 指标

主指标是 64 局的平均 `score`，也就是 0 到 1 之间的分数，作用和 WebShop 的 mean reward 一样。辅助指标有这几个：成功率；按任务类型分别统计的平均 score；错误 focus 的局数占比；平均步数；`in_valid` 为 false 的步数占比；解析失败的步数占比；自动消歧的次数。

已发表的结果里，小模型的成功率基本贴地板，所以选配置、比较各臂都看 score，不看成功率。

## 8. 失败标签和接入 runner

### 8.1 失败标签

写 `envs/sciworld/failure_labels.py`。它只读轨迹、每步 info 和环境结构，不调用模型。

逐步标签按下表打。

| 条件 | 步标签 |
|---|---|
| 返回 `No known action matches that input.` | `invalid_action` |
| 触发了歧义菜单 | `ambiguous` |
| 动作以 `focus on` 开头，并且之后分数变成 -100 | `wrong_focus` |
| 分数比上一步高 | `progress` |
| 其余情况 | `no_progress` |

每局再给一个主标签，按下面的优先级取第一个命中的。

| 优先级 | 标签 | 条件 |
|---|---|---|
| 1 | `success` | 成功 |
| 2 | `wrong_focus` | 整局出现过 `wrong_focus` |
| 3 | `invalid_dominant` | `invalid_action` 加上解析失败的步数超过总步数一半 |
| 4 | `loop` | 最后 6 步里只有不超过 2 种不同动作 |
| 5 | `stalled_partial` | 最终分数大于 0，并且最后 10 步分数没有变化 |
| 6 | `stalled_zero` | 最终分数为 0 |
| 7 | `timeout_partial` | 其余未成功的局 |

另外，每局结束时调用 `get_goal_progress()`，把它的输出解析成子目标列表，每项包含名称、类别（有序、无序或可选）和是否完成。记录第一个没完成的有序子目标的名称，字段名 `first_unmet_subgoal`。`get_goal_progress()` 的返回格式请先在 1.2.3 里确认，再写解析代码，并在报告里贴一个解析前后的例子。

还要记录 gold 覆盖率，即这一局对应变体的 gold 动作序列里，有多少比例的动作在 agent 的动作里原样出现过，不考虑顺序。

### 8.2 接入 runner

在 env 注册表里加 `sciworld`，`run.py` 的现有参数都要能用。解码沿用 Visibility 报告的 S1 设置：agent 是 Qwen3.5，vLLM 服务；thinking 打开；`max_tokens` 4096；`EVOLVER_AGENT_TEMPERATURE=0`；top_p 0.8；history_length 2。

轨迹写到 `R{N}/trajectories/{episode_id}.md`，frontmatter 和 ALFWorld 一致，另外加 `task_name`、`variation`、`score`、`episode_label` 和 `first_unmet_subgoal`。

给 kev 的固定格式渲染放在 `envs/sciworld/render.py`，函数名 `render_for_reader(traj) -> str`，和 TextCraft 那份保持一致。只包含这几部分：任务描述；每一步的动作和环境原始返回；每步的分数；最后的结果。不包含 agent 看到的 prompt。

## 9. 检查

这几项依次过，前一项不过不要往下做。

G0，任务选择可复现。在两个不同的进程里分别跑 `select_tasks.py` 和切分脚本，`task_selection.json` 和 `splits.json` 的 sha256 都要相同。

G1，环境正确。对 held-in 和 held-out 全部 128 局，按对应变体的 gold 动作序列重放，至少 95% 的局拿到 100 分。没拿到的局逐个列出来，写明原因。再写一个随机 agent，从当前合法动作列表里均匀抽取，每局跑 2 次，报告平均 score 和成功率，作为 floor。

G2，标签正确。抽 20 局失败的 baseline 轨迹，人工核对整局标签，报告和人工判断一致的比例。低于 0.85 的话，先修规则。

G3，和已发表数字对照。已发表的 prompt 设置下，Qwen2.5-7B 在类似的 easy 设置、30 步上限下，score 在 17 到 28 之间，成功率在 6 到 12 之间。我们的模型、步数和任务子集都不同，不要求对齐。但如果 Qwen3.5-4B 在 held-in 上的平均 score 低于 0.05 或高于 0.70，先贴出 5 条完整轨迹查原因，不要往下做。

G4，baseline。用 4B、2B、0.8B 在 held-in 和 held-out 上各跑一次 R0。我希望看到 4B 的 held-in 平均 score 在 0.10 到 0.60 之间。不在的话，报告实际数字和按任务类型的分解，不要自行调整任务选择。

## 10. 交付物

1. 新增的代码文件和 `VENDOR.md`。
2. `task_selection.json`、`splits.json`，以及它们的 sha256。
3. 一份报告 `SCIWORLD_BASELINE.md`，包含这些内容：G0 到 G4 的结果表；三个模型在 held-in 和 held-out 上的平均 score、成功率、按任务类型的 score；整局标签的分布；`first_unmet_subgoal` 最常见的 10 个；随机 agent 的 floor；每步平均耗时；每个模型一条得分最高和一条失败的完整轨迹。
4. 每局一条 JSON 记录，放在 `runs/sciworld_baseline/`，所有数字都能从这些记录重算。

## 11. 运维和凭证

token、API key、任何密钥都不能出现在报告、日志、轨迹 dump 和 commit 里。key 只从环境变量读取。日志里打印配置时要把含 `key`、`token`、`secret` 的字段打码。提交前用 grep 扫一遍这几个关键词。JVM 进程在 worker 退出时必须关掉，报告里写明跑完以后没有残留的 java 进程。
