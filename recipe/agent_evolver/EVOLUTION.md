# Harness-JEV 文本协议进化器：完整方法与结果

> 目标模型 Qwen3.5-4B · meta 模型 deepseek-v4-pro（mioffice）· 只进化 harness，**模型权重永不训练**。
> 本文档记录 `recipe/agent_evolver/` 的完整实现方法与截至 2026-10-01 的全部结果。

---

## 0. 摘要（TL;DR）

- **进化引擎 100% 复用 harnessx 原生的 `harnessx.meta_harness.MetaAgent.evolve()`，一字未改。**
- 唯一分叉：**被进化的 rollout** 不是 harnessx 原生 tool-calling runloop，而是我们自己的 `<think>/<action>` 文本 ReAct loop —— 因为 vLLM 服务没起 `--enable-auto-tool-choice --tool-call-parser`，原生 tool-calling 跑不通。
- `HarnessConfig` YAML 在这里是**纯载体**：只放两个 no-op processor（system prompt + guidance + 5 个 loop 标量），供 meta-agent 当 YAML 读/改；真正的 rollout 用 `spec_from_config()` 反解回 `HarnessSpec` 再跑文本 loop。
- **结果（64-task，temp 0.4 采样）**：ALFWorld 从裸基线 held-in 0.359 / held-out 0.281，进化到 held-in **0.703** / held-out **0.750**（≈翻倍），且 held-out ≥ held-in = **无过拟合、真泛化**。WebShop 从裸基线 0.203 (mean 0.516) 进化到 pass **0.359** / mean_reward **0.635**。

---

## 1. 目标与边界

对 Qwen3.5-4B 做**纯 harness 进化**：只进化 system prompt / 每轮 guidance / loop 参数，模型权重永不训练。进化在 `held-in` 上做，`held-out` 是泛化探针——held-in 涨但 held-out 不涨 = prompt 过拟合，两个都涨 = 真泛化。

**可进化面（收窄且锁定）**：

| 面 | 载体 | 说明 |
|---|---|---|
| 系统提示 | `SystemPromptProcessor` 内联 `prompt:` | 基线 = 裸 Qwen 默认 |
| 每轮 guidance | `TextEvolverSpecProcessor.guidance` | 注入每轮 user prompt 的 `{skills}` 槽，**主杠杆** |
| `loop_breaker_k` | 同上 | 连续 k 步重复同一动作时屏蔽它 |
| `history_mode`/`history_n`/`history_obs_clip` | 同上 | 历史如何渲染给模型 |
| `tried_action_mask` | 同上 | 是否隐藏已试过的动作 |

其余 harnessx 能力（原生 processors、tool_registry 等）**全部不暴露**——防止 meta-agent 去做描述级泄露的改动（见记忆 [[meta-actionspace-isolation]]）。

---

## 2. 与 harnessx 的关系

**方法 = harnessx 原生。** 我们复用的正是 tau2/tb2/gaia 三个 recipe 的同一契约：

> recipe 自持 rollout + eval + gate + round 循环，每轮调用 `meta_agent.evolve(current_config, trajectories_dir, output_dir)`，进化产物是一个 `HarnessConfig` YAML。

| 层 | 实现 | 是否 harnessx 原生 |
|---|---|---|
| 进化引擎 | `harnessx.meta_harness.MetaAgent.evolve()` | ✅ 一字未改 |
| post-flight 校验 | `harnessx.meta_harness.validate_workflow.EvolveValidator` | ✅ 一字未改 |
| 回归门 | `recipe/agent_evolver/gate.py:score_and_gate` | ✅ 照抄 tau2 `_score_and_gate` |
| 被进化的 rollout | `recipe/agent_evolver/agents.py`（文本 ReAct） | ❌ 我们自己的 |

---

## 3. 系统架构与数据流

```
                    ┌─────────────────────────────────────────────┐
                    │  meta-agent（强模型 deepseek-v4-pro）        │
                    │  工具: Read/Write/Edit/Glob/Grep/Bash/       │
                    │        WebSearch/WebFetch/spawn_reflect_worker│
                    │  技能: analyze/journal/reference/validate     │
                    │        + alfworld/webshop-playbook            │
                    │  约束: WriteScopeGate/ReadScopeGate/          │
                    │        LeakageGuard/ContractAutoCheck/...     │
                    └───────────────▲─────────────────┬────────────┘
                       读轨迹+当前config │                 │ 写新 config.yaml
                                        │                 ▼
        ┌───────────────────────────────┴─────────────────────────────────┐
        │  round 循环（recipe/agent_evolver/run.py，recipe 自持）           │
        │   R{N}/config.yaml ──► run_text() rollout ──► 打分 ──► gate       │
        │          ▲                                            │          │
        │          └──────── meta_agent.evolve() + journal 归因 ─┘          │
        └──────────────┬────────────────────────────────────────┬─────────┘
                       │ spec_from_config 解码                    │ rollout
                       ▼                                          ▼
        ┌──────────────────────────┐        ┌────────────────────────────────┐
        │  agent（弱模型 Qwen3.5-4B）│        │  env servers（setsid nohup 常驻）│
        │  vLLM 8200-8203          │ ──►    │  ALF held-in  18082-85 (140 games)│
        │  LiteLLMProvider, temp0.4 │  POST  │  ALF held-out 18086-89 (134 games)│
        │  <think>/<action> ReAct  │  /step │  WebShop 18090 (12087 goals)      │
        └──────────────────────────┘        └────────────────────────────────┘
```

**一轮的完整流程**（`run.py` 主循环）：

1. `current_config.to_yaml_file(R{N}/config.yaml)` —— 把当前 config 落盘。
2. `run_text(config, benchmark, task_ids, api_bases, env_urls, seed, concurrency)` —— 按 api_base 分片、并发 rollout，每 task 得到 `{task_id, success, reward, steps, goal, trajectory, final_obs}`。
3. 写每 task 轨迹 `R{N}/trajectories/{task_id}.md`。
4. 打分：`passed = Σ success`，`avg_reward = mean(reward)`。
5. `score_and_gate(...)` —— 回归门（见 §4.5）。
6. 若 `round_idx ≥ 1`：journal 归因回填（`compute_attribution` + `fill_gating`）。
7. 若非最后一轮：`await meta_agent.evolve(current_config=R{N}/config.yaml, trajectories_dir=R{N}/trajectories, output_dir=R{N+1}/evolve)`；产物 `R{N+1}/evolve/config.yaml` 经 `canonicalize()` 后成为下一轮 `current_config`。
8. 冷却 60s（meta 后端会突发大量请求，避开限流窗口）。

---

## 4. 进化机制详解

### 4.1 可进化面：config 是纯载体

config YAML 只含两个 processor（见 `native/config.py`）：

```yaml
processors:
  - _target_: harnessx.processors.context.system_prompt.SystemPromptProcessor
    system_builder:
      _target_: file://…/native/prompt.py::StaticSystemPromptBuilder
      prompt: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
  - _target_: file://…/native/text_spec.py::TextEvolverSpecProcessor
    guidance: ""              # ← 主杠杆：每轮注入的 {skills}
    loop_breaker_k: 3
    history_mode: raw
    history_n: 2              # 进化中被 meta 改成 8
    history_obs_clip: 150
    tried_action_mask: false
```

- `SystemPromptProcessor` + `StaticSystemPromptBuilder`：把 `prompt:` 内联成系统提示。
- `TextEvolverSpecProcessor`：**no-op 载体**（继承 `MultiHookProcessor` 但不挂任何 hook），只是把 6 个标量当构造参数存着，供 meta-agent 当 YAML 字段编辑。
- 两者在原生 runloop 里都不会产生任何行为 —— 这让 `validate_workflow` 的 replay gate 用合成任务跑一遍新 config 时「能跑通但不崩」。

**反解** `spec_from_config(cfg) -> HarnessSpec`：遍历 processors，抽出 `prompt:` 和 `TextEvolverSpecProcessor` 的 6 个字段，重组成 `HarnessSpec(system_prompt, guidance, processors={loop_breaker, history_format, tried_action_mask})`。

### 4.2 rollout 层（被测量的东西）

`agents.run_alfworld` / `run_webshop` 是 Skill1 `<think>/<action>` 文本 ReAct loop（模板逐字取自 verl-agent，保证基线复现其 step-0 成功率）。

**ALFWorld**（`agents.py:171`）：
1. `POST /reset {game_idx, seed}` → 初始 `observation` + `admissible_commands`。
2. 循环直到 `done` 或 `steps ≥ 50`：
   - `processors.mask_admissible(history, adm, spec.processors)` 过滤动作（loop_breaker / tried_action_mask）；
   - `processors.render_history(history, ...)` 渲染历史；
   - 用模板拼 user prompt（首步用 `*_NO_HIS`，之后带历史），`{skills}` 槽 = `spec.guidance`；
   - `provider.complete([system, user])` → Qwen3.5-4B（temp 0.4 / top_p 0.8 / max_tokens 1024 / `enable_thinking=False`）；
   - `extract_action` 取 `<action>…</action>`（小写；malformed 则取 raw 尾部当非法动作交给 env 惩罚，**不强制回退**）；
   - `POST /step {action}` → 新 obs + admissible + `won` + `done`。
3. `success = won`，`reward = 1.0 if won else 0.0`。

**WebShop**（`agents.py:251`）：`POST /create` 建 uuid 实例，循环到 `done` 或 `steps ≥ 15`，`reward` 是连续值（0..1），`success = reward ≥ 0.999`，结束 `POST /destroy`。

### 4.3 进化引擎 `MetaAgent.evolve()` 内部（`harnessx/meta_harness/agent.py:538`）

每次调用做四件事：

1. **准备 brief**：把当前 config 路径 + 轨迹目录写进 `scratch/TASK.md`（meta-agent 的任务书）；若 `memo_path`（learnings.md）已有 journal，再生成 `scratch/CONTEXT.md`（最近 N 轮的上下文）。
2. **组 meta-agent 自己的 harness**（`build_meta_agent_harness_config`）：工具 Read/Write/Edit/Glob/Grep/Bash/WebSearch/WebFetch + `spawn_reflect_worker`；processor 链 = SystemPrompt + WriteScopeGate + StepDeadlineReminder + CostGuard + LoopDetection + ToolResultNoiseFilter + Compaction；`SOUL.md` 人设 + `analyze/journal/reference/validate` 四组内置技能 + 我们注入的 `alfworld/webshop-playbook`。
3. **跑 meta-agent**：`harness.run(BaseTask(description=TASK.md))`，带 wall_clock（3600s）和成本上限（$50）；meta-agent 读轨迹、写出一份新的 `output_dir/config.yaml`。
4. **post-flight**（`EvolveValidator.run`，见下）。

### 4.4 post-flight 校验（`harnessx/meta_harness/validate_workflow.py`）

`EvolveValidator` 三阶段：

- **validity（阻塞）**：`canonicalize`（config 可规范化）→ **replay gate**（用合成任务 "Reply OK" 跑一遍新 config，保证它不崩——对我们的 no-op 载体恒通过）。
- **policy（结构 diff 非空时阻塞）**：`novelty`（新颖性）+ `evidence`（改动有轨迹证据）。
- **advisory（永不阻塞）**：literals 检查。

`compute_changeset` 做两份 canonical config 的浅 diff，供 novelty/evidence/journa 使用。

### 4.5 回归门（`recipe/agent_evolver/gate.py`，照抄 tau2）

`score_and_gate(round_reward, round_cost, round_idx, round_config, best, tolerance, cost_weight)`：

- 首轮：`accept` 并 establish baseline。
- 后续：`adjusted = round_reward - cost_weight·max(cost 增量, 0)`；若 `adjusted ≥ best_adjusted - tolerance`（默认 tolerance=0.02）→ `accept`；否则 `reject` 并**回退** `current_config` 到 `best_cfg`。

> ⚠️ **WebShop 特例**：`round_reward = mean_reward`（连续 0..1），**不是 pass rate**。所以 R1 虽然 pass 从 14→11 掉，但 mean_reward 0.4348→0.4365 微涨，门仍 accept。ALFWorld 的 reward 是二值（won→1.0），所以 mean_reward ≡ pass rate，两者一致。

### 4.6 journal 归因（`harnessx/meta_harness/journal.py`）

每轮（round_idx≥1）回填：`compute_attribution(entry.predicted_affected, passed_now, passed_before, ...)` 算出每个受影响 task 的 `flipped/still_F/still_T/regressed`，`fill_gating` 记录 outcome（accepted/reverted）+ `regressed_unpredicted`。这些成为下一轮 `CONTEXT.md` 的输入，让 meta-agent 知道「上一刀改对了没、哪些 task 被误伤」。

### 4.7 并发与 sharding（`run.py:run_text`）

- `n_workers = min(len(api_bases), concurrency)`，task_ids 按 worker 数切连续片。
- worker i 用 `api_bases[i]` + `env_urls[i % len(env_urls)]`，**串行**处理自己那片的 task。
- **ALFWorld env server 是单实例 stateful**，一个 worker 不能在同一 server 上重叠 episode → 4 held-in / 4 held-out server 与 4 个 api_base 一一配对。
- **WebShop** 每 task 一个 uuid `env_id`（contextvar 隔离），单 server 18090 可同时吞 4 个 worker。
- `_post_retry`：`/reset`（幂等）带重试；`/step` **故意不重试**（动作可能已服务端执行，重试会 double-step）。

---

## 5. 模块清单

### harnessx（引擎，未改）

| 模块 | 作用 |
|---|---|
| `harnessx.meta_harness.MetaAgent`（`agent.py:493`） | `evolve()` 主入口 |
| `harnessx.meta_harness.agent.build_meta_agent_harness_config`（`agent.py:98`） | 组 meta-agent 自己的 harness |
| `harnessx.meta_harness.agent.compute_changeset`（`agent.py:439`） | 两份 canonical config 浅 diff |
| `harnessx.meta_harness.validate_workflow.EvolveValidator` | post-flight 三阶段校验 |
| `harnessx.meta_harness.replay.run_replay_gate_strict` | replay 门 |
| `harnessx.meta_harness.journal` | `read_entries`/`build_context`/`compute_attribution`/`fill_gating` |
| `harnessx.meta_harness.processors.*` | WriteScopeGate / ReadScopeGate / LeakageGuard / ContractAutoCheck / StepDeadlineReminder / ToolResultNoiseFilter |
| `harnessx.core.harness.HarnessConfig` | 载体 YAML（`to_yaml_file`/`from_yaml_file`/`canonicalize`） |
| `harnessx.core.model_config.ModelConfig` | 包 provider（`agentic()`） |
| `harnessx.core.processor.MultiHookProcessor` | `TextEvolverSpecProcessor` 基类 |
| `harnessx.providers.anthropic_provider.AnthropicProvider` | **meta 后端**（deepseek-v4-pro，anthropic SDK） |
| `harnessx.providers.litellm_provider.LiteLLMProvider` | **agent 后端**（Qwen3.5-4B，vLLM） |
| `harnessx.processors.context.system_prompt.SystemPromptProcessor` | 承载 system prompt |

### recipe/agent_evolver（我们的薄适配层）

| 模块 | 作用 |
|---|---|
| `run.py` | round 循环：rollout → gate → evolve → journal 回填 |
| `agents.py` | `run_alfworld`/`run_webshop`（文本 ReAct rollout）+ `extract_action` |
| `native/config.py` | `make_*_baseline_config` + `spec_from_config`（encode/decode） |
| `native/text_spec.py` | `TextEvolverSpecProcessor`（no-op 载体） |
| `native/prompt.py` | `StaticSystemPromptBuilder` |
| `spec.py` | `HarnessSpec` dataclass + 常量（temp 0.4 / top_p 0.8 / max_steps 50/15） |
| `gate.py` | `score_and_gate`（照抄 tau2） |
| `processors.py` | `mask_admissible` / `render_history`（loop 规则） |
| `trajectories.py` | 写 per-task `.md` |
| `skills/{alfworld,webshop}-playbook/SKILL.md` | 给 meta-agent 的 benchmark 攻略（失败模式 + lever 映射 + 红线） |

---

## 6. 运行方法

### 6.1 infra（`benchmarks/start_jev_infra.sh`，幂等）

```bash
bash benchmarks/start_jev_infra.sh
```

- vLLM ×4 `:8200-8203`（Qwen3.5-4B，GPU 0..3，`--gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 256 --dtype bfloat16`）
- ALFWorld held-in ×4 `:18082-18085`（`ALFWORLD_SPLIT=eval_in_distribution`，140 games，game_idx 0-63）
- ALFWorld held-out ×4 `:18086-18089`（`ALFWORLD_SPLIT=eval_out_of_distribution`，134 games，game_idx 0-63）
- WebShop ×1 `:18090`（`WEBSHOP_NUM_PRODUCTS=full WEBSHOP_HUMAN_GOALS=1 WEBSHOP_SEED=233`，12087 goals）

全部 `setsid nohup` 常驻。注意：机器重启 / supervisord 周期动作会清掉 vLLM 和 WebShop，需重跑此脚本。

### 6.2 进化 CLI（`recipe/agent_evolver/run.py`）

```bash
# 基线复测（1 轮，无进化）
./.venv/bin/python -m recipe.agent_evolver.run alfworld --num-rounds 1 --num-tasks 64
./.venv/bin/python -m recipe.agent_evolver.run webshop  --num-rounds 1 --num-tasks 64

# held-in 进化（3 轮 = 1 基线 + 2 次进化）
./.venv/bin/python -m recipe.agent_evolver.run alfworld --num-rounds 3 --num-tasks 64 --run-tag heldin-evolve-v1

# 从已有 config 续跑（held-in 二次进化 / held-out 泛化探针）
./.venv/bin/python -m recipe.agent_evolver.run alfworld --split heldin  --num-rounds 3 --base-config .../heldin-evolve-v1/R2/config.yaml --run-tag heldin-evolve-v2
./.venv/bin/python -m recipe.agent_evolver.run alfworld --split heldout --num-rounds 1 --base-config .../heldin-evolve-v2/R2/config.yaml --run-tag heldout-probe-v2r2
```

关键参数：`--split {heldin,heldout}`、`--num-tasks`、`--seed`（ALF 默认 1234 / WS 默认 0）、`--num-rounds`、`--base-config`、`--run-tag`、`--meta-model`/`--meta-api-base`（默认 `anthropic/volcengine_maas/deepseek-v4-pro`）、`--evolve-cost`/`--evolve-steps`/`--evolve-wall-clock`、`--regression-tolerance`（默认 0.02）。

---

## 7. 全部结果

> 单位：64-task 采样（game_idx/task_id 0-63），agent temp 0.4 → 单次 run 噪声 ±5-6 个百分点（ALF 二值）；WebShop 看连续 mean_reward。

### 7.0 运行总览（9 run / 17 轮）

| run | split | 轮数 | 状态 |
|---|---|---|---|
| baseline-alf-heldin | held-in | 1 | ✅ |
| baseline-alf-heldout | held-out | 1 | ✅ |
| baseline-ws-test | WebShop test | 1 | ✅ |
| heldin-evolve-v1 | held-in | 3 | ✅ |
| heldout-probe-r2 | held-out | 1 | ✅ |
| heldin-evolve-v2 | held-in | 3 | ✅ |
| heldout-probe-v2r2 | held-out | 1 | ✅ |
| webshop-evolve-v1 | WebShop test | 3 | ✅ |
| webshop-evolve-v2 | WebShop test | 3 | ✅ |

ALFWorld 与 WebShop 各 **6 个进化 rollout 轮**（v1 3 轮 + v2 3 轮）。每个 N 轮 run = **N 次 rollout + (N−1) 次进化**（meta-agent pass）。2-task 冒烟（`smoketest`/`smoke2`）不计入。

每个 N 轮 run = **N 次 rollout + (N−1) 次进化**（meta-agent pass）。2-task 冒烟（`smoketest`/`smoke2`）不计入。下面 7.1–7.6 是逐 run 逐轮明细。

### 7.1 基线复测（裸 Qwen system + 空 guidance）

| split | passed | pass_rate | mean_reward |
|---|---|---|---|
| ALF held-in | 23/64 | 0.359 | 0.3594 |
| ALF held-out | 18/64 | 0.281 | 0.2812 |
| WebShop test | 13/64 | 0.203 | 0.5164 |

（与记忆锚点 [[jev-baseline-3way-anchor]] ALF 0.344/0.344、WebShop 0.281/r0.535 一致，噪声内，无崩塌。）

### 7.2 ALFWorld held-in 进化 v1（从裸基线，3 轮）

| 轮 | passed | pass_rate | 门 |
|---|---|---|---|
| R0 基线 | 21/64 | 0.328 | establish |
| R1 进化 | 18/64 | 0.281 | **reject → 回退**（meta 写了过冗长 guidance） |
| R2 进化 | 29/64 | **0.453** | **accept（+12.5pt）** |

R2 把 guidance 蒸馏成 3 条高信号规则 + "推理 1-2 短句"；`history_n` 2→8。

### 7.3 held-out 泛化探针 v1（v1-R2 config 跑 held-out）

| 结果 | passed | pass_rate |
|---|---|---|
| held-out + v1-R2 | 39/64 | **0.609**（裸 held-out 0.281 → +32.8pt） |

### 7.4 ALFWorld held-in 进化 v2（从 v1-R2 续跑，3 轮）

| 轮 | passed | pass_rate | 门 |
|---|---|---|---|
| R0（复测 v1-R2） | 31/64 | 0.484 | establish |
| R1 进化 | 27/64 | 0.422 | **reject → 回退** |
| R2 进化 | 45/64 | **0.703** | **accept（+22pt）** |

R2 关键突破：meta **自己摸出了 ALFWorld 的精确动作词表**（见 §8），把 guidance 从 3 条扩到 5 条，直接压掉 malformed-action 失败。

### 7.5 held-out 泛化探针 v2（v2-R2 config 跑 held-out）

| 结果 | passed | pass_rate |
|---|---|---|
| held-out + v2-R2 | 48/64 | **0.750**（≥ held-in 0.703 → 无过拟合） |

### 7.6 WebShop 进化 v1（从裸基线，3 轮）

| 轮 | passed | pass_rate | mean_reward | 门 |
|---|---|---|---|---|
| R0 基线 | 14/64 | 0.219 | 0.4348 | establish |
| R1 进化 | 11/64 | 0.172 | 0.4365 | **accept**（mean_reward 微涨 ≥ best−tol） |
| R2 进化 | 23/64 | **0.359** | **0.6349** | **accept（mean +0.20）** |

> WebShop 门优化的是连续 `mean_reward`（见 §4.5），所以 R1 pass 掉但 mean 涨仍 accept；R2 则是 mean 大跳（0.4348→0.6349）且 pass 翻倍（0.219→0.359）。
> R2 guidance 摸出了 WebShop 评分机制：**商品页要 click 每一个匹配的属性选项（color/size/scent/flavor/quantity/package count）才被选中，商品标题不能替代点击**；短 query 搜索；搜不到就进最接近商品的 options 里找属性，不翻页、不重复搜。

#### WebShop 进化 v2（从 v1-R2 续跑，3 轮）

| 轮 | passed | pass_rate | mean_reward | 门 |
|---|---|---|---|---|
| R0（复测 v1-R2） | 20/64 | 0.312 | 0.6285 | establish |
| R1 进化 | 15/64 | 0.234 | 0.5417 | ❌ reject → 回退 |
| R2 进化 | 11/64 | 0.172 | 0.4862 | ❌ reject → 回退 |

> **饱和**：两轮进化都回归被回退，best 停在 v1-R2（mean ~0.63）。被拒的 R2 尝试是给 guidance 加「检查商品标题匹配非可点击属性（material/fit/sole type/certification），标题与指令矛盾就别买」——思路对，但整体 guidance 过度复杂化，pass 反而掉到 0.172。对比 ALFWorld 第 4–6 轮（v2）仍从 0.484 涨到 0.703，WebShop 在第 3 轮就摸到了评分语法、之后进入边际递减/反噬区。

### 7.7 汇总

| split | 裸基线 | 进化后 | Δ |
|---|---|---|---|
| ALFWorld held-in | 0.359 | **0.703** | +0.344 |
| ALFWorld held-out | 0.281 | **0.750** | +0.469 |
| WebShop test | 0.203 (mean 0.516) | **0.359 (mean 0.635)** | +0.156 pass / +0.12 mean |

> WebShop 进化为 v1-R2 最佳（v2 两轮回退，未再提升，见 §7.6）。

### 7.8 逐轮规律

- **门每次都把「进化做砸」的轮回退**：v1-R1（0.281）、v2-R1（0.422）都是 reject，incumbent 没被带崩。
- **每次回退后，下一轮 meta 都能做出更优**：v1 R1(0.281) → R2(0.453)；v2 R1(0.422) → R2(0.703)。
- **泛化单调变好、从未掉头**：held-out 0.281 → 0.609（v1）→ 0.750（v2）。
- WebShop 复刻了同一模式：R1 pass 掉（14→11）但 mean 微涨（0.4348→0.4365）仍 accept；R2 mean 大跳至 0.6349 且 pass 翻倍至 0.359（meta 摸出「click 每个属性选项才算选中」的评分规则）。
- **WebShop 第 4–6 轮饱和、ALFWorld 第 4–6 轮仍涨**：WebShop v2 两轮（0.542/0.486）都回归被回退，best 停在 v1-R2；ALFWorld v2-R2（第 6 轮）从 0.484 涨到 0.703。

### 7.9 WebShop synthetic held-in 进化（synthetic=held-in / human=held-out 协议）

> 方法学纠正后的正确协议：合成目标（env 18091，11.67M）做 held-in 进化，人类目标（env 18090，12087）只做 held-out 测试。以下是 held-in 侧全部 10 轮（v1 3 轮 + v2 3 轮 + v3 4 轮）。§7.6 的人类池结果为「同池内泛化」参考，非真 held-out。

**synthetic 基线比 human 硬 ~3.4×**：synthetic 裸基线 mean 0.154（pass 0.047），human 裸基线 mean 0.516（pass 0.203）—— 真 OOD gap 坐实。

**v1（从裸基线，3 轮）**

| 轮 | passed | mean_reward | 门 |
|---|---|---|---|
| R0 基线 | 3/64 (0.047) | 0.1541 | establish |
| R1 进化 | 4/64 (0.062) | 0.4759 | accept（mean ×3） |
| R2 进化 | 0/64 (0.0) | 0.0415 | ❌ reject（崩） |

R1 摸出 synthetic 的同一「购买协议」（提属性 → 短 query → 点精确 color/size → Buy Now），mean 三倍。R2 加「描述词匹配」规则（goal.py 的 `num_attr_matches` 计数标题/描述里的材质/风格词）过度复杂化，mean 崩到 0.0415 被回退。

**v2（从 v1-R1 续跑，3 轮）**

| 轮 | passed | mean_reward | 门 |
|---|---|---|---|
| R0（复测 v1-R1） | 5/64 (0.078) | 0.5184 | establish |
| R1 进化 | 4/64 (0.062) | 0.4128 | ❌ reject |
| R2 进化 | 6/64 (0.094) | 0.4630 | ❌ reject |

两轮都回归被回退，best = v1-R1 购买协议（mean 0.5184）。

**v3（从 v1-R1 续跑，4 轮）**

| 轮 | passed | mean_reward | 门 |
|---|---|---|---|
| R0（复测 v1-R1） | 1/64 (0.016) | 0.4795 | establish |
| R1 进化 | 5/64 (0.078) | 0.3070 | ❌ reject |
| R2 进化 | 3/64 (0.047) | 0.4719 | ✅ accept（0.4719 ≥ 0.4795−0.02 贴线） |
| R3 进化 | 0/64 (0.0) | 0.1169 | ❌ reject（崩） |

R2 是唯一一次「accept」但 0.4719 仍 < R0 0.4795（贴 2pt 容差线，非严格更好），R3 又崩到 0.1169 被回退。**best 仍是 v1-R1**，10 轮下来没有一个后续轮严格超过它。

**关键观察**：

1. synthetic 上「pass」（reward≥0.999 精确命中）几乎不涨（0.047→0.094），但「mean reward」（部分分）三倍（0.154→0.518）—— 合成模板目标极难精确命中，guidance 的价值全在 mean。
2. synthetic 天花板（mean ~0.52）明显低于 human 池（mean 0.635），合成目标本身更难/OOD。
3. v1-R1 与 v2-R0 同一 config 跨 run 的 mean 0.476→0.518 是 vLLM bf16/重启噪声（比典型 ±2-4pt 大，注意）。
4. **10 轮饱和诊断**：v1-R1 在第 1 轮就摸到购买协议、触及 ceiling（mean ~0.48-0.52，跨 3 次复测 0.476/0.518/0.479 都在噪声带），后面 9 轮（v1-R2、v2-R1/R2、v3-R1/R2/R3）没有一个严格超过它——要么过度复杂化崩掉（v1-R2 0.0415、v3-R3 0.1169），要么贴线 accept（v3-R2 0.4719 < 0.4795）。meta-agent 在 synthetic 上已无增量，瓶颈不是「没试够」，是「这个 benchmark 的可迁移杠杆在第 1 轮就挖干了」。

**human held-out 迁移探针（2026-10-01，env 18090, task 0-63, seed 0）**：取 best synthetic config（v1-R1 购买协议）跑 human held-out，结果 **pass 15/64 (0.234) / mean_reward 0.5469**，对比 human 裸基线 **pass 0.203 / mean 0.516**：

| | pass | mean_reward |
|---|---|---|
| human 裸基线 | 0.203 | 0.516 |
| human + synthetic best | 0.234 | 0.5469 |
| Δ | **+3.1pt** | **+3.1pt** |

**判定：synthetic→human 不迁移。** synthetic 上三倍的赢面（mean 0.154→0.518）到 human 只剩 +3.1pt，落在 vLLM bf16/重启噪声带（±2-4pt，本 memory 观测到过 ±4pt）内 —— 净中性。购买协议 guidance 是拟合 synthetic 模板目标（`instruction_attributes` 笛卡尔积生成）的，人类自然语言目标吃不到这口。**WebShop 与 ALFWorld 的关键分野**：ALFWorld 的 meta 发现的是任务无关的动作词表（held-in→held-out 真泛化 0.359→0.750），WebShop synthetic 进化摸到的却是 synthetic 模板的评分语法，跨 pool 不通用。真 OOD 探针把 WebShop 的「突破」判为假象。

### 7.10 WebShop 官方 train→test 对齐（2026-10-01）

> 用户要求放弃 synthetic/human，改用**官方 train/test**：train（human goals 1500–12086，取 goal 1500–1563）做 meta 进化，test（goals 0–500，取 goal 0–63）只测。这是 ALFWorld「held-in 进化 / held-out 测试」的 WebShop 对应物 —— 同一批人类自然语言目标池的两个切片，同分布不同实例（不像 synthetic 是跨分布）。

**干净起点**：test 裸基线 pass 0.250 / mean 0.509；train R0 裸基线 pass 0.188 / mean 0.4715 —— 同量级、噪声带内（≈ALFWorld held-in≈held-out）。

**train 进化 3 轮**（`ws-train-evolve-v1`）：

| 轮 | pass | mean_reward | 门 |
|---|---|---|---|
| R0 基线 | 12/64 (0.188) | 0.4715 | establish |
| R1 进化 | 7/64 (0.109) | 0.3421 | ❌ reject |
| R2 进化 | 12/64 (0.188) | **0.5560** | ✅ accept（+8.4pt） |

R2 摸到的 guidance 是**通用购物协议**：短 query（产品类型+最重要属性）→ 只开第一页 promising 商品 → **click 每个 size/color 选项一次（点了即选中，页面文字不变也算）** → 匹配就 buy now、缺属性换商品绝不重开/重搜、没候选就买最好的别死循环。第 3 条是 WebShop 的 benchmark 级评分机制（之前人类池 R2 也独立摸到过），第 4 条反 thrash —— 都不是 synthetic 模板特异的。

**test 迁移探针**（best=R2 跑 test）：

| | pass | mean_reward |
|---|---|---|
| test 裸基线 | 0.250 | 0.509 |
| test + train R2 | 0.219 | 0.5594 |
| Δ | −3.1pt | **+5.0pt** |

**判定：弱正、需确认。** train 上 +8.4pt 搬到 test 剩 +5.0pt（~60% 迁移），与 synthetic→human 的 ~0% 形成对照 —— 说明「通用购物协议」比「synthetic 模板语法」可迁移。但 +5pt 只是刚过 ±2-4pt 噪声带（test 历史 mean 0.46–0.55，本次 0.559 是历史最高），且 pass 反向 −3.1pt 不确认。**需 2-3 seed 或 128/256 task 复测才能把「真迁移」和「噪声」分开**。

**128-task matched A/B 复测（2026-10-01，同 seed 0 背靠背）**：

| 128 task | pass | mean_reward |
|---|---|---|
| 裸基线 | 27/128 (0.211) | 0.4936 |
| train-R2 config | 31/128 (0.242) | 0.5676 |
| Δ | +3.1pt | **+7.4pt** |

**判定：坐实弱正迁移。** mean +7.4pt（0.494→0.568）明显过噪声带，且与 64-task 的 +5pt 方向一致、跨样本量复现 —— 真迁移。pass +3.1pt 转正但仍在一个标准误内（±5pt），不显著。**结论定性**：WebShop 官方 train→test 对齐后，harness 进化**能**产生可迁移增益，但增益全在 mean（部分分）、不在 pass（不显著）—— meta 摸到的「通用购物协议」让弱模型离满分更近，但没能让它真正解出更多单。与 ALFWorld 的「动作词表→pass 翻倍」、synthetic→human 的「零迁移」是三种不同的结局。

---

## 8. 关键发现

1. **进化真泛化、无过拟合**：ALFWorld 两 split 从 ~0.3 涨到 ~0.7+，且 held-out（0.750）≥ held-in（0.703）。因为 meta 发现的是**任务无关的 ALFWorld 动作语法**，不是 held-in 特有记忆。
2. **meta 自摸动作词表**（v2-R2 的核心）：
   - 目标解码两步：「先改状态再搬去 Y」——`put a hot/clean X in Y` 里 Y 只是终点，不是改状态设备；
   - 固定搜索顺序：先台面（countertop/sidetable/…）再容器，`examine` 关着的容器没用，不重访已知空处；
   - 精确动作形式：`take X from Y`（不许裸 `take X` / `pick up`）、`heat X with microwave`、`cool X with fridge`、`clean X with sinkbasin`、先 `go to` 再 `move X to Y`。
   这正是此前 `invalid_actions`（malformed action）失败的主因——4B 模型不知道确切动作词表。
3. **门有效**：v1-R1、v2-R1 都是 meta 过度发挥导致回归，门 reject + 回退，incumbent 得以守住；下一轮 meta 从失败归因中恢复并做出更优 config。
4. **系统 prompt 始终没动**（裸 Qwen 默认）：全部增益来自 `guidance` + `history_n`，说明对 4B 弱模型，**每轮可执行的指令 > 全局身份设定**。
5. **跨 benchmark 同构突破**：WebShop R2 复刻了 ALFWorld v2-R2 的同一模式——meta 摸出了该 benchmark 的**评分机制**（`click` 每个匹配的属性选项才算选中，标题不能替代点击；短 query；进 options 找属性），mean_reward 0.4348→0.6349、pass 0.219→0.359。两个 benchmark 上都是「meta 发现任务评分语法 → 弱模型照着做 → 分涨」。
6. **饱和点不同**：ALFWorld 到第 6 轮仍在涨（v2-R2 0.484→0.703）；WebShop 在第 3 轮（v1-R2）摸到评分语法后即饱和，第 4–6 轮 meta 的进一步规则（加「标题匹配非可点击属性」）过度复杂化、两轮全回退。提示「评分语法」这类高杠杆规则一旦命中，后续收益骤降、反噬风险上升。

---

## 9. 已知坑 / 红线

- **meta 后端**：`AnthropicProvider` 必须 `extended_thinking=False`（mioffice fork 无 `temperature` 参数，开 thinking 会 TypeError）；`max_tokens ≥ 4096` 否则 thinking 吃光预算。（[[harnessx-jev-strong-meta-backend]]）
- **LiteLLM 启动延迟**：`LITELLM_LOCAL_MODEL_COST_MAP=True` 必须在 `import litellm` **之前**设（否则每进程 ~45s 拉远程 cost map）。
- **env 并发**：ALFWorld server 单实例 stateful，worker 必须每 server 串行、每 worker 独立端口；WebShop 每 task 一个 uuid。（[[harnessx-djev-env-sharding]]）
- **`/step` 不重试**：动作可能已服务端执行，重试会 double-step。
- **infra 会被清**：机器重启 / supervisord 周期动作会清掉 vLLM 和 WebShop，报 `httpx.ConnectError: All connection attempts failed` 时先重跑 `start_jev_infra.sh`。
- **跨 run 绝对数值有 ±2-4pt 的 vLLM bf16/重启噪声**，逐 seed 的 gate 只在 run 内可比。（[[harnessx-jev-opened-o-negative]]）

---

## 10. 下一步

- [x] **WebShop held-in/held-out**：synthetic held-in 10 轮（v1 3 + v2 3 + v3 4）+ human held-out 探针均已跑完。**判定：synthetic→human 不迁移**（human +3.1pt 落噪声带，见 §7.9）；且 synthetic 第 1 轮就饱和（后 9 轮无一严格超 v1-R1）。WebShop 的「突破」是真 OOD 探针下的假象 —— 进化摸到的是 synthetic 模板评分语法，非任务无关语法。
- [ ] WebShop 六轮（人类池）已跑完、第 3 轮后饱和（best=v1-R2 mean 0.635）；需在官方 test 另开一组独立 seed / 更大样本（128/256 task）复测，确认 0.359/0.635 非噪声。
- [x] 对 ALFWorld v2-R2 与 WebShop v1-R2 各做 held-out / 独立 seed 泛化探针，坐实「真泛化」（ALFWorld 已坐实 0.750≥0.703；WebShop 已验，**真 OOD 下不迁移**，见 §7.9）。
- [ ] 若两 benchmark 都稳，则结论「harness 进化对弱模型是任务无关的通用增益」成立。
- [ ] 可探：WebShop 饱和后，用更强 meta 后端或「失败轨迹定向注入」看能否突破天花板（参照 [[harnessx-beam-search-result]] 的饱和诊断：瓶颈在探索广度非 gate）。

---

## 11. 相关文档

- [META_AGENT_DECISIONS.md](META_AGENT_DECISIONS.md) — meta-evolve 里 LLM 的 14 个决断清单、各落在哪个 module、JEV 可替代性三档（执行层已决定论化 / 诊断-选杠杆层可封顶式替代 / LLM 本质层替代即失价值）。

---

## 12. 模型规模扫描（4B / 2B / 0.8B × ALFWorld / WebShop）

> 同一 text 协议进化器换 agent 底座（`EVOLVER_AGENT_MODEL`），ALF held-in 进化 6 轮→held-out 探针；WS train（`--start 1500`）进化 6 轮→test（`--start 0`）探针。探针 base=进化里 mean_reward 最高轮。run-tag：`{alfheldin,alfheldout,ws-train,ws-test}-{2b,0.8b}-v1`。4B 为历史 run（heldin-evolve-v2 / ws-train-evolve-v1）。

**总表**（pass 率；WS 括号内 mean_reward）：

| 模型 | ALF held-in 基线→best | ALF held-out | WS train 基线→best | WS test |
|---|---|---|---|---|
| 4B | 0.484 → **0.703** (R2) | 0.750 | 0.188(0.472) → 0.188(**0.556**, R2) | 0.219(0.559) / 128-task 0.242(0.568) |
| 2B | 0.109 → **0.250** (R3) | 0.219 | 0.141(0.550) → 0.250(**0.646**, R4) | 0.297(0.656) |
| 0.8B | 0.000 → **0.078** (R3) | 0.000 | 0.000(0.016) → 0.125(**0.394**, R4) | 0.109(0.343) |

**三条结论**：

1. **绝对分数随规模单调坍缩**：ALF best 0.703 / 0.250 / 0.078；ALF 基线 0.484 / 0.109 / 0.000。但 WS 基线 mean 2B 0.550 反而略高于 4B 0.472 —— 连续 reward 对弱模型「宽容」（弱模型靠行为协议拿部分分）。

2. **二值 vs 连续 reward 决定「能力地板」是否致命**：
   - ALFWorld（二值 success）：0.8B 卡 0 三轮，第 3 轮才 0.078 且 **held-out 归零** —— 二值任务有硬能力门槛，模型走不通动作-状态闭环时 harness 零杠杆。
   - WebShop（连续 reward）：0.8B 基线 mean 0.016 → best 0.394（**+37.8pt / ×24**），test 0.343 保留 87% —— 只要有部分分可拿，harness 进化在地板模型上仍显著有效。

3. **相对增益随规模下降而升，绝对增益反之**：WS mean Δ 4B +8.4 / 2B +9.5 / 0.8B +37.8pt；ALF pass Δ 4B +21.9 / 2B +14.1 / 0.8B +7.8pt。弱模型基线越低，行为协议能撬的百分比空间越大，但绝对天花板仍低。

**运行学**：2B/0.8B evolve 全程 0 crash（deepseek meta 后端稳定）；evolve 耗时逐轮膨胀（2B ALF 6.4→11.6→14.3min，meta 读的轨迹/learnings 累积）；全失败轨迹（0.8B ALF 前三轮）无成功信号可学，收敛更慢（0.8B 第 3 轮才破零，2B 第 3 轮到 0.250，4B 第 2 轮就翻倍）。

相关：[[harnessx-jev-model-matrix]]（memory 完整版）、[[harnessx-jev-evolve-v2-actiongrammar]]（4B ALF）、[[harnessx-jev-webshop-evolve]]（4B WS）。
