# archive/rule_arm — A3「rule 注入臂」归档

精简 Harness-JEV（Phase 2a）：把 A3 三臂里的 **rule 注入臂**（用确定性 A2 规则 tagger 当先验塞进 meta brief 的通道）整线归档，主线只留 ALF+WS 的 kev 三轴（llm / prior / enforce）。

## 归档内容

| 文件 | 原位置 | 作用 |
|---|---|---|
| `rule_prior.py` | `recipe/agent_evolver/` | `build_rule_priors()`：把 `rule_tagger` 输出写成 `decision_priors.md`，喂给 meta 的同一 brief 通道（`--decision-backend rule` 分支调用）。 |
| `a3_three_arm_3seed.sh` | 仓库根 | A3 三臂 × 3 seeds 的 runner（llm / kev / **rule**）。A3 已停（scale 扫描优先），本归档连带退役。 |

## 同步改动

- `run.py`：删 `--decision-backend rule` 分支（`build_rule_priors` 调用），`--decision-backend` 现在只认 `llm`（默认）或注册的 jevlike 决策模型名（如 `kev`）。

## 为什么 `rule_tagger.py` 留在主线

`rule_tagger.py`（`alfworld_rule` / `gridgames_rule`）是 A2 **确定性规则 tagger**，是 A1/A2 gate GA 的 rule baseline 打分原语，仍被未归档的 `recipe/agent_evolver/a1a2_runner.py`（`from .rule_tagger import ...`）引用——A1/A2 报告（`A1A2_RESULTS.md` + REPORT §2）复现靠它。它依赖 `failure_labels.py`（同样留在主线）。

被归档的 `rule_prior.py` 里的 `from .rule_tagger import ...` 是冻结快照、不要求原地可跑；复活 = `git mv` 回 `recipe/agent_evolver/`。
