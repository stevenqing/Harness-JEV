# WebShop 程序化失败标签（`recipe/agent_evolver/webshop_labels.py`）

对应 `specs/kev_ranker_step0c.md` §4.2。六个标签，按顺序判定，取第一个成立；`reward == 1` 不是失败，不打标签。

## reward 分解的取法

WebShop 的 reward 由四部分构成，全部在环境 `web_agent_site.engine.goal.get_reward` 里、购买时（`buy now`）算好：

| 分量 | 语义 | 取值 |
|---|---|---|
| `r_type` | 购买商品的类型是否匹配 goal 类型 | 0 / 1 |
| `r_att` | goal 属性命中的比例 | 0..1 |
| `r_option` | goal 选项命中的比例（无选项时为 `None`） | 0..1 或 `None` |
| `r_price` | 价格是否 ≤ goal 价格上限（无上限时为 `None`） | 0 / 1 或 `None` |

总 reward = `(num_attr_matches + num_option_matches + r_price) / (len(attributes) + len(options) + 1) * r_type`，因此 `reward == 1` ⟺ 四项全满。

环境原实现里这四个分量存进了 `user_sessions[session_id]['verbose_info']` 但没透给 HTTP 调用方。改动分两处，**算法不变，只是把已经算好的分量透传出来**：

1. `benchmarks/webshop/env_server.py` 的 `POST /step` 在 `done` 时把该 session 的 `verbose_info` 作为 `reward_info` 字段返回（`step` 在 done 时会 `reset()` 换新 session id，所以在 `step` 前先记下旧 session id 再读）。
2. `recipe/agent_evolver/agents.py` 的 `run_webshop` 在 done 时把 `reward_info` 存进轨迹 record。

没有从 goal 属性列表和购买页自己重算匹配——那需要复刻 `get_type_reward` / `get_attribute_reward` / `get_option_reward` 的 fuzzy 匹配，容易和环境不一致；直接用环境已经算好的分量，保证和 reward 标量自洽。

## 判定顺序

| 标签 | 条件 |
|---|---|
| `no_search` | 轨迹里没有任何 `search[...]` 动作 |
| `not_bought` | 搜过，但在步数内没有 buy（无 `reward_info`） |
| `wrong_type` | 买了，`r_type == 0` |
| `attr_mismatch` | 买了，类型对，`r_att < 1` |
| `option_mismatch` | 买了，类型和属性对，`r_option < 1` |
| `price_over` | 买了，前三项都对，`r_price == 0` |

防御分支：四项全满但 `reward < 1`（理论上不会发生）归 `other`。浮点用 `>= 1 - 1e-6` 判"满"。

## kev 的问题

kev 对 WebShop 失败轨迹回答一道六选一（`ChooseQ`），选项就是上面六个标签，描述见 `failure_types.py::WS_QUESTION`。
