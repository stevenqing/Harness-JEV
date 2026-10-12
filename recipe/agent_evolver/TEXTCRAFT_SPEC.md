# SPEC：TextCraft 接入 agent_evolver

写给 coding agent。目标是把 TextCraft 接进 `recipe/agent_evolver`，让它和 ALFWorld、WebShop 用同一套 runner、同一套解码、同一种 prompt 形状、同一种轨迹格式。本 spec 只做接入和 baseline，不跑进化，不训练任何模型。第 9 节的检查全部通过、baseline 报告交给我看过之后，才进入进化实验。

## 0. 范围

要做的事情有六件。第一，把环境以进程内调用的方式接进来。第二，生成一份固定的任务清单，并切出 held-in 和 held-out。第三，写 prompt 模板，结构和 `agents.py:ALFWORLD_TEMPLATE` 一致。第四，写动作解析。第五，写一个从环境结构推出失败标签的模块。第六，用 Qwen3.5 的 4B、2B、0.8B 各跑一次 R0 baseline。

不做的事情有三件。不跑 meta 进化。不改 kev。不碰 ALFWorld、WebShop、gridgames 现有的任何代码路径，新代码只能新增文件，或者在 env 注册表里加一项。

## 1. 环境来源和安装

环境代码来自 AgentGym，仓库 https://github.com/WooooDyy/AgentGym ，commit `3ef9235`，目录 `agentenv-textcraft/agentenv_textcraft/`。只要其中三样东西：`environment.py`、`crafting_tree.py`、`recipes/` 目录（860 个 JSON）。把它们放进 `recipe/agent_evolver/envs/textcraft/vendor/`，保留原仓库的 LICENSE 文件，并在同目录写一个 `VENDOR.md`，记下来源 URL、commit 和改动列表。

不要用 AgentGym 的 FastAPI server。它是单进程，每步都会打印，`/commands` 和 `/goal` 两个接口调用的函数并不存在，`reset` 里还调用了全局的 `random.seed(42)`，多线程 reset 会互相干扰。我们在 rollout worker 里直接 import `TextCraftEnv`，进程内调用。

不要用 ADaPT 原版（archiki/ADaPT）的环境代码。它的 `traverse_recipe_tree` 会原地扩展配方列表，每 reset 一次配方树就变大一些，大约 90 次以后会卡住。AgentGym 已经重写了这个函数，没有这个问题。

依赖只有 gymnasium，不需要 Java 或 Docker。transformers 不是真正需要的依赖，不要为它改环境。一棵 `CraftingTree` 可以被所有 env 实例共享，建一次大约 0.04 秒。

## 2. 固定任务清单

原代码里 task 编号到目标物品的对应关系依赖 `os.listdir` 的顺序，干扰配方的抽样依赖 `PYTHONHASHSEED`，换一台机器、换一个进程，同一个编号就是另一道题。所以我们不在 reset 时现场生成题目，而是先生成一份固定清单，reset 时只从清单里读。

写 `envs/textcraft/build_manifest.py`，输出 `envs/textcraft/manifest.jsonl`，每行是一个 task，字段如下。

| 字段 | 内容 |
|---|---|
| `task_id` | 形如 `tc_0000`，按下面的排序规则编号 |
| `goal` | 目标物品名，用人类可读形式，比如 `golden carrot` |
| `depth` | `tree.get_min_depth(goal)` |
| `gold_recipes` | `tree.traverse_recipe_tree(goal)` 给出的配方，每条是 `{output, count, ingredients:[{item,count}]}`，有多种配方的物品全部列出 |
| `distractor_recipes` | 干扰配方，结构同上 |
| `commands_text` | 环境展示给 agent 的完整配方文本，原样保存 |
| `base_items` | gold 配方里所有不可合成、只能 `get` 的原料 |

排序规则要和文件系统无关。先取所有 depth 至少为 1 的可合成物品，按 `(depth, goal)` 排序，goal 按字符串比较。每个 task 的干扰配方用 `random.Random(hash_stable(task_id))` 生成，`hash_stable` 用 sha256 的前 8 个字节转整数，不能用 Python 内置的 `hash`。抽样之前先把候选列表排好序，不要直接从 set 里抽。

生成完以后，在两个不同的 `PYTHONHASHSEED` 下各跑一次，两份 manifest 的 sha256 必须相同，这是第 9 节的 G0。

### 2.1 切分

从 manifest 里切出两组，互不重叠，每组 64 个 task，按 depth 分层抽样，数量如下。

| depth | held-in | held-out |
|---|---|---|
| 1 | 12 | 12 |
| 2 | 26 | 26 |
| 3 | 22 | 22 |
| 4 | 4 | 4 |

这样分配的原因是，已发表的小模型数据显示 depth 1 接近满分，depth 4 接近 0，有用的信号主要在 depth 2 和 3。抽样用固定种子，结果写进 `envs/textcraft/splits.json`，字段是 `heldin` 和 `heldout` 两个 `task_id` 列表。

另外再抽一组 `compat100`，按 depth 31、41、25、3 抽 100 个，和 AgentGym 的 AgentEval 测试集的深度构成一致，只用来做第 9 节 G3 的对照检查，不参与进化。注意它和 AgentEval 的具体物品并不相同，因为原编号在我们机器上不可复现。

## 3. 环境包装

写 `envs/textcraft/env.py`，提供和现有 ALFWorld、WebShop 包装相同的接口，即 `reset(task_id) -> (obs, info)` 和 `step(action_text) -> (obs, reward, done, info)`。

reset 时从 manifest 读出目标和 `commands_text`，用这份文本初始化环境，不调用原代码里的随机生成。初始 observation 是 `commands_text` 加一行 `Goal: craft <goal>.`。

### 3.1 observation 规范化

环境返回的文字里会出现 `minecraft:oak_planks` 这种 id，小模型会照抄回来，导致下一步命令无法解析。包装层把所有 `minecraft:xxx_yyy` 改写成 `xxx yyy`。`craft()` 往 stdout 打印的 `Wrong Item Count` 要捕获并丢弃，不能出现在日志里。

### 3.2 可选动作列表

环境本身不提供可选动作列表，需要我们合成，规则如下。第一条固定是 `inventory`。然后把 `commands_text` 里展示的每条配方都写成一条完整的 craft 命令，格式是 `craft <count> <output> using <n1> <ingredient1>, <n2> <ingredient2>`，数量照配方填。最后，对展示的配方里出现的每个不可合成原料，生成一条 `get <n> <item>`，n 取配方里的数量。配方里是泛称的原料（例如 `planks`），展开成配方集合里出现过的具体物品。

这个列表只是展示给模型的提示，不做硬约束。模型输出的命令即使不在列表里，也照样交给环境执行，由环境判断对错。每一步在 info 里记下 `in_admissible` 这个布尔值。

是否展示这个列表，是一个 harness 变量，配置名 `show_admissible`，baseline 里默认为 true，和 ALFWorld 的做法一致。

### 3.3 结束条件和分数

合成出目标物品时成功，reward 为 1，done 为 true。没有 done 动作，所以靠步数上限结束，`max_steps` 设为 30，和 AgentGym-RL 的评测协议一致。连续失败提前结束这条规则 baseline 里不开，作为 harness 变量 `early_stop_after_failures` 保留，默认为 None。

info 里每一步都要记下这几样东西：`inventory`（dict）、`env_message`（环境原始返回）、`in_admissible`、`step_idx`。

### 3.4 已知的环境行为

这些行为不要去改环境，但都要在 `VENDOR.md` 里写明，也要在轨迹 dump 里能看出来。

| 行为 | 说明 |
|---|---|
| 只认精确数量 | `craft 5 leather using 20 rabbit hide` 会被拒绝，只能一次合成一份，重复执行 |
| 输出数量被忽略 | `craft 9 leather using 4 rabbit hide` 只合成 1 个，环境不报错 |
| `get` 没有资源限制 | 想拿多少都行 |
| `get` 一个可合成物品或泛称会失败 | 返回 `Could not find X` |
| 大小写敏感 | `Craft` 开头会失败，末尾带句号也会失败 |
| 合成干扰配方会成功 | 环境不报错，但对目标没有帮助 |

## 4. prompt 模板

在 `agents.py` 里新增 `TEXTCRAFT_TEMPLATE`，结构和 `ALFWORLD_TEMPLATE` 对齐。system 用 Qwen3.5 默认 system，不要沿用 AgentGym-RL 里写死的 Qwen2.5 system 文本。

```
You are an expert agent crafting items in a Minecraft-like text environment. Your task is to: craft {goal}.
Here are the crafting commands you can use:
{commands_text}
Rules: always give a quantity, e.g. "get 1 lapis lazuli" or "craft 1 golden carrot using 8 gold nugget, 1 carrot". Do not say whether an item comes from your inventory or the environment. Use only the recipes given above.
{skills}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
```

`Rules:` 那一行就是 AgentGym 的三条提醒，保留它们是为了让 baseline 能和 AgentGym-RL 的数字大致对得上。`{skills}` 是 guidance 槽，baseline 为空，进化时由 meta 填。`history_length` 默认是 2，和 ALFWorld 一致。`show_admissible` 为 false 时，整行 `Your admissible actions ...` 不出现。

历史里放不放 `<think>` 的内容，按 ALFWorld 现有的做法来，只放动作和 observation，不放 think。

## 5. 动作解析

写 `envs/textcraft/projection.py`。从输出里取最后一对 `<action>...</action>` 之间的文本。然后依次做四件事：去掉首尾空白，把连续空格合并成一个，把开头的动词转成小写（只限 `get`、`craft`、`inventory` 三个），去掉末尾的句号。

没有 `<action>` 标签，或标签里是空的，记为解析失败。这一步不调用环境，返回固定的 observation `Invalid action format.`，并消耗一步。

规范化前后的文本都记进 info，字段名分别是 `raw_action` 和 `action`，这样能统计规范化到底起了多少作用。

## 6. 指标

主指标是成功率，也就是 64 个 task 里合成出目标的比例。另外按 depth 分别报告成功率。辅助指标有三个：平均步数；`in_admissible` 为 false 的步数占比；解析失败的步数占比。

## 7. 失败标签

写 `envs/textcraft/failure_labels.py`，作用和 ALFWorld 的 `failure_labels.py` 相同。它只读轨迹、manifest 和每步的 info，不调用模型。输出给 A1/A2 打分用，也给后面"从环境结构推出失败类型"用。

### 7.1 逐步标签

每一步按环境返回的消息打标签。

| 环境消息 | 步标签 | 怎么细分 |
|---|---|---|
| `Could not execute <a>` | `format` | 无 |
| `Wrong item format: X` | `format` | 无 |
| get 之后返回 `Could not find X` | `get_invalid` | 再分三种。X 在 gold 树里可合成，记 `get_craftable`。X 是泛称，记 `get_tag`。其余情况记 `get_unknown_form`，一般是单复数或多了标点 |
| craft 之后返回 `Could not find enough items to craft X` | `craft_missing` | 把配方和当前库存做差，再分三种。缺的是某个可合成的中间物，记 `missing_intermediate`。缺的是原料，记 `missing_base`。种类齐但数量不够，记 `wrong_count`。配方本身不在 manifest 里，记 `hallucinated_recipe` |
| craft 之后返回 `Could not find a valid recipe for ...` | `craft_invalid` | 输入凑齐了但不对应任何配方，多数是数量写错或想批量合成 |
| craft 成功，但产物不在 gold 树里 | `crafted_distractor` | 无 |
| 其余成功的 get 或 craft | `ok` | 无 |

### 7.2 整局标签

每局给一个主标签，按下面的优先级取第一个命中的。

| 优先级 | 标签 | 条件 |
|---|---|---|
| 1 | `success` | 成功 |
| 2 | `format_dominant` | `format` 步占全部步数一半以上 |
| 3 | `loop` | 最后 6 步里只有不超过 2 种不同动作 |
| 4 | `never_attempted_goal` | 整局从没尝试过 craft 目标物品 |
| 5 | `stuck_intermediate` | 最后一次失败的步标签是 `missing_intermediate` 或 `get_craftable` |
| 6 | `stuck_count` | 最后一次失败的步标签是 `wrong_count` 或 `craft_invalid` |
| 7 | `timeout_other` | 其余未成功的局 |

同时输出每局各类步标签的计数，以及 gold 树里有多少比例的中间物最终出现在库存里，后者是一个 0 到 1 的进度值。

## 8. 接入 runner

在 env 注册表里加 `textcraft`，`run.py` 的现有参数都要能用。解码沿用 Visibility 报告的 S1 设置：agent 是 Qwen3.5，vLLM 服务；thinking 打开；`max_tokens` 4096；`EVOLVER_AGENT_TEMPERATURE=0`；top_p 0.8；history_length 2。

轨迹文件写到 `R{N}/trajectories/{task_id}.md`，frontmatter 字段和 ALFWorld 一致，另外加上 `depth` 和 `episode_label`。

还要给 kev 写一份固定格式的轨迹渲染，不能直接用 agent 看到的 prompt。原因是 harness 一改，prompt 就变，kev 的输入分布会跟着漂移。固定格式只包含这几部分：`Goal: craft X`；配方列表；每一步的动作、环境原始返回、库存变化；最后的结果。函数名 `render_for_reader(traj) -> str`，放在 `envs/textcraft/render.py`，ALFWorld 和 WebShop 以后也会改成同样的做法。

## 9. 检查

这几项依次过，前一项不过不要往下做。

G0，清单可复现。在两个不同的 `PYTHONHASHSEED` 下分别生成 manifest 和 splits，sha256 相同。

G1，环境正确。写一个 gold agent，按 gold 配方的拓扑顺序依次执行 get 和 craft。它在 held-in 和 held-out 全部 128 个 task 上的成功率必须是 100%。再写一个随机 agent，从可选动作列表里均匀抽取，每个 task 跑 4 次，报告成功率作为 floor。

G2，标签正确。抽 20 局失败的 baseline 轨迹，人工核对整局标签，报告和人工判断一致的比例。低于 0.85 的话，先修规则再往下做。

G3，和已发表数字对照。用 Qwen3.5-4B 在 `compat100` 上跑一次，报告整体和分 depth 的成功率。AgentGym-RL 里 Qwen3-4B 是 45，分 depth 是 87、37、12、0。我们的 prompt 和模型都不同，不要求对齐，但如果整体低于 20 或高于 75，先把 5 条完整轨迹贴出来查原因，不要往下做。

G4，baseline。用 4B、2B、0.8B 在 held-in 和 held-out 上各跑一次 R0。我希望看到 4B 的 held-in 成功率在 0.10 到 0.60 之间。如果不在，报告实际数字和分 depth 结果，不要自行调整切分。

## 10. 交付物

1. 新增的代码文件和 `VENDOR.md`。
2. `manifest.jsonl`、`splits.json`，以及它们的 sha256。
3. 一份报告 `TEXTCRAFT_BASELINE.md`，包含 G0 到 G4 的结果表；三个模型在 held-in 和 held-out 上的总成功率和分 depth 成功率；整局标签的分布；随机 agent 的 floor；每个模型一条成功和一条失败的完整轨迹。
4. 每局一条 JSON 记录，放在 `runs/textcraft_baseline/`，所有数字都能从这些记录重算。

## 11. 运维和凭证

token、API key、任何密钥都不能出现在报告、日志、轨迹 dump 和 commit 里。key 只从环境变量读取。日志里打印配置时要把含 `key`、`token`、`secret` 的字段打码。提交前用 grep 扫一遍这几个关键词。
