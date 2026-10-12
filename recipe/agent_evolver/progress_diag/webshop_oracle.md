# WebShop 进度 oracle：reward 分解 → 卡位序号

本文是进度诊断 SPEC §4.3 的 WebShop 半边。**oracle 已经存在**（`recipe/agent_evolver/webshop_labels.py`），这里只补两件事：把六路标签重排成「进度序号」，并指出它作为进度 oracle 的粗粒度局限。

## 1. 已有的 oracle（不要重写）

`webshop_labels.py::ws_label(record)` 是 WebShop 的程序化 ground-truth，六路标签按 first-match-wins 判定（完整算法 + reward 公式见同目录 `webshop_labels.md`）。关键点：

- reward 四分量 `r_type / r_att / r_option / r_price` 由环境在 **buy 时**算好（`goal.get_reward`），`env_server.py` 在 `POST /step` done 时透传成 `reward_info`，`agents.py::run_webshop` 存进 record。
- 没 buy 的轨迹**没有** `reward_info`，只有动作序列（`trajectory[].action`）可读。

## 2. 六路标签 → 进度卡位序号 m

`ws_label` 的标签顺序是**决策顺序**（first-match-wins），不是进度顺序。进度诊断里 oracle 的「卡位序号 m」=「第一个未达成的里程碑」，必须按下表右列重排：

| ws_label | 语义 | 卡位序号 m | 里程碑 |
|---|---|---|---|
| `no_search` | 没发过 `search[...]` | 1 | 搜到商品 |
| `wrong_type` | 买了，`r_type == 0` | 2 | 点进类型匹配商品 |
| `attr_mismatch` | 类型对，`r_att < 1` | 3 | 属性匹配 |
| `option_mismatch` | 类型+属性对，`r_option < 1` | 4 | 选项匹配 |
| `price_over` | 前三项对，`r_price == 0` | 5 | 价格不超限 |
| `not_bought` | 搜过但没 buy（无 `reward_info`） | 6 | 完成购买 |

> `not_bought` 在 `ws_label` 里排第 2，是决策顺序（先判断「没 buy」再判断「买了但哪不对」）；**进度顺序里它是最后一步**（搜了、卡在没买成）。别把两者混了。

## 3. 关键局限：`not_bought` 的粗粒度

`reward_info` 只在 buy 时产生，所以「搜了但没买」的失败轨迹只有一个 `not_bought` 标签，**无法**区分它实际卡在 milestone 2/3/4/5 的哪一档——除非从轨迹动作序列自己重算属性/选项/价格匹配。`webshop_labels.py` 刻意不这么做（要复刻 `get_attribute_reward`/`get_option_reward` 的 fuzzy 匹配，容易和环境不一致）。

后果：WebShop 的 oracle 卡位分布可能天然集中在 milestone 6（`not_bought`）。这不是 oracle 的 bug，是 reward 只在 buy 点结算的结构性事实。对 G3 分布 gate 的修正见 `align.md` §4。
