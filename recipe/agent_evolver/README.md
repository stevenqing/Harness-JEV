# Harness-JEV：只进化 harness 的 agent 进化器（+ kev System-1 先验注入）

> **核心主张：换 harness，不换模型。** agent 权重**永不训练**——被进化的只有 `system_prompt` / `guidance` / 历史窗口 / loop 参数；一个冻结的 System-1 判别读头（kev）给强 meta-agent 注入「失败机制」先验。
>
> agent 模型 = **Qwen3.5-4B**（冻结，vLLM 服务）· meta-agent = **deepseek-v4-pro**（读轨迹、写 guidance）· kev = **Qwen3.5-4B-Base + LoRA + pointer head**（冻结读头，一次前向打分）。
>
> 本文档讲清**方法 + 如何复现**。方法/结果的全量详述见 [EVOLUTION.md](EVOLUTION.md)（纯 harness 进化）、[KEV_THREE_AXIS.md](KEV_THREE_AXIS.md)（三轴 kev 先验）、[META_AGENT_DECISIONS.md](META_AGENT_DECISIONS.md)（meta 决断清单）。仓库级总表见根 [RESULTS.md](../../RESULTS.md) 与 [REPORT_visibility.md](../../REPORT_visibility.md)。

---

## 0. TL;DR

用同一套文本协议进化器（`run.py`）做两件事：

1. **纯 harness 进化**：meta-agent 读失败轨迹、改 `guidance`/loop 参数，把 Qwen3.5-4B 在 ALFWorld 上从 held-in **0.359 → 0.703**、held-out **0.281 → 0.750**（≈翻倍，零过拟合）。
2. **kev 三轴先验注入**（本 repo 的增量贡献）：一个冻结的 System-1 读头在 **一次前向** 里给失败轨迹打出三轴（lens × lever × intent）概率向量，作为**软先验**喂给 meta-agent——held-out 上 ALFWorld **.844**（vs 纯 llm .781、硬 enforce .594）、WebShop **.630**（vs .560 / .567）。

**三轴对照结论（`llm` vs `prior` vs `enforce`）**：

| benchmark | baseline | llm | **prior** | enforce |
|---|---|---|---|---|
| ALFWorld（held-out pass） | .297 | .781 | **.844** | .594 |
| WebShop（held-out mean_reward） | .439 | .560 | **.630** | .567 |

> **软先验 > 硬裁决 > 不做先验**：kev 的弱读给强 meta 保留覆盖权就赢（prior）；做成绑定 gate 反而砍伤搜索空间（enforce）。

---

## 1. 方法总览

### 1.1 只进化 harness（不碰权重）

每一轮（`run.py` 主循环）：

```
R0 裸基线 ──rollout 64 task──▶ 写 trajectories ──gate──▶ meta_agent.evolve() ──▶ R1 config
                                                                                  │
   R1 rollout ──gate──▶ evolve ──▶ R2 ...   （gate: mean_reward ≥ best − tolerance，否则回退 best）
```

- **进化载体是 `HarnessConfig` YAML**：只放两个 no-op processor——`SystemPromptProcessor`（系统提示）+ `TextEvolverSpecProcessor`（`guidance` 文本 + `history_n`/`loop_breaker_k`/`history_mode`/`history_obs_clip`/`tried_action_mask` 6 个 loop 标量）。meta-agent 把它当纯 YAML 读/改。
- **meta-agent = 强模型**读每轮轨迹、写下一轮 guidance。它发现的「任务语法」（ALFWorld 的精确动作词表、WebShop 的逐属性点击）是增益的真正来源——系统提示全程未动。
- **held-in 进化 / held-out 只测**：ALF 在 `eval_in_distribution`（140 可解局）进化 6 轮，`eval_out_of_distribution`（134）只测一次；WebShop 用官方 train 切片（`--start 1500`）进化、test 切片（`--start 0`）只测。held-in 涨而 held-out 不涨 = prompt 过拟合，两个都涨 = 真泛化。

### 1.2 kev 三轴先验注入（System-1 → System-2）

kev 是一个**冻结的判别式读头**：一次前向读出「失败机制」的概率向量，不生成、不训练。它替 meta-agent（System-2）做那个「读 50+ 条轨迹、归因失败模式」里最便宜、最稳定、最抗偏置的一段。

**三轴词表**（benchmark-agnostic，定义在 `decision_prior.py`）：

| 轴 | 问题 | 取值 |
|---|---|---|
| **lens**（看什么） | 是否被 blocker 卡住 / 是否能力不足 | `lens_failure` / `lens_capability_gap` |
| **lever**（改什么） | 修复归到哪个杠杆 | `configuration` / `control` / `action` / `instruction` |
| **intent**（意图） | candidate 会不会把失败翻正 / 把成功翻挂 | `corrective` / `preservative-transfer` / `preservative-lock` |

**两种注入模式**（`--decision-mode`）：

| 模式 | lens | lever | intent | 后端挂掉时 |
|---|---|---|---|---|
| `prior`（软先验） | 注入 brief | 注入 brief（可覆盖） | 不启用 | warn + 回退 llm |
| `enforce`（硬绑定） | 注入 brief | **绑定 + gate 检查** | **candidate retrocheck**（`pass_prob<0.5` 拒绝该轮） | 硬 fail（loud abort） |

数据流：`rollout 失败轨迹 → kev 一次前向打分 → decision_priors.md → 注入 meta brief`（prior）；`enforce` 再把 lever 决策做成绑定 gate、把 intent 做成 candidate 级反事实校验。

> **为什么弱读反而赢**（机制，见 [REPORT §10](../../REPORT_visibility.md)）：meta 的瓶颈不是「能否理解」，而是「以什么认知成本/方差/偏置理解」。kev 判别式逐条读 logit 再聚合，天然把 base rate 数对、把失败模式抽象成可泛化的语义——soft prior 喂给强模型是正则化；硬 gate 越线让弱模型做合成级裁决，反而砍空间。

---

## 2. 结果速览

### 2.1 三轴对照（temp=0 确定性，64 task，6 轮 + held-out 探针）

**ALFWorld**（held-out = `eval_out_of_distribution` 探针，各臂 held-in best config 重放）：

| arm | held-in best（轮） | held-out | 泛化 Δ |
|---|---|---|---|
| baseline | .344（R0） | .297 | −.047 |
| llm | .734（R5） | .781 | +.047 |
| **prior** | .672（R5） | **.844** | **+.172** |
| enforce | .531（R3） | .594 | +.063 |

**WebShop**（held-out = 官方 test 切片）：

| arm | held-in best | held-out（pass / mean_reward） |
|---|---|---|
| baseline | — | .203 / .439 |
| llm | .281 / .536（R3） | .312 / .560 |
| **prior** | .312 / .598（R4） | **.344 / .630** |
| enforce | .250 / .562（R5） | .203 / .567 |

### 2.2 kev 先验的历史对照（Mode A，find/acquire 词表）

同一软先验、benchmark 专属词表时代（已被三轴取代）：ALFWorld held-out **llm .797 → kev .969**（+17pt）。

### 2.3 kev reviewer 的准确率锚点（A1/A2 gate GA）

kev 当「考官」判 ALF 失败轨迹的 `find/acquire`：准确率 **.672** vs 规则 tagger .352（**+0.32**），Cohen's κ 0.398。语义失败模式（找/取目标）是 kev 的增量所在；state-readable 机械模式（gridgames 掉坑/loop）上 kev 是**负增量**（规则 5 行 `has_loop` 完胜）。

### 2.4 规模扫描（2B / 0.8B 三轴）

同一三轴先验、只换 agent 底座（kev reviewer 不变）：

| cell | baseline | llm | prior | enforce |
|---|---|---|---|---|
| 2B ALF held-out | .047 | .141 | **.172** | .125 |
| 2B WS held-out（reward） | .483 | .504 | **.537** | .440 |
| 0.8B ALF | 全 0.000（二值地板，无失败模式可标注） | — | — | — |

**kev 相对增益随规模缩**：4B 上 prior−llm ≈ +.06~.07，2B 只剩 +.03，掉进 64-task 噪声。enforce 持续最弱。

> 纯 harness 进化（无 kev）的模型矩阵与 WebShop 迁移、gridgames decoding 根因，见 [EVOLUTION.md](EVOLUTION.md) 与根 [RESULTS.md](../../RESULTS.md)。

---

## 3. 目录结构

```
recipe/agent_evolver/
├── run.py                  # 进化器主循环（rollout + gate + evolve + journal + 三轴 decision 注入）
├── spec.py                 # HarnessSpec 数据类 + 环境变量默认值
├── gate.py                 # 回归门 score_and_gate（accept/revert）
├── agents.py               # <think>/<action> 文本 ReAct rollout（被进化的东西）
├── trajectories.py         # per-task .md 轨迹写入
├── decision_prior.py       # 三轴词表（lens×lever）+ ALF/WS 轨迹 parser/renderer + DecisionPriors
├── candidate_retrocheck.py # enforce 的 intent 轴：candidate 级反事实校验（pass_prob<0.5 拒绝）
├── reviewer.py             # A1/A2 gate GA 的 kev reviewer 词表（find/acquire）+ 指标
├── failure_labels.py       # 失败模式 ground-truth 标签（A1/A2 用）
├── rule_tagger.py          # A2 确定性规则 tagger（A1/A2 rule baseline 用）
├── a1a2_runner.py / a1a2_aggregate.py   # A1/A2 逐轨迹打分 + 汇总（复现 REPORT §2）
├── stats.py / baseline.py  # 采样统计 / 基线
├── native/                 # HarnessConfig 构造 + processor + prompt builder
├── skills/                 # meta-agent 的 playbook skill（alfworld / webshop）
├── multi_node/             # run_multinode.sh + verify_meta_pod.sh（跨节点/cloudml）
├── runs/evolve/<tag>/      # 每次 run 的产物：R{n}/config.yaml + trajectories/ + comparison.json
├── EVOLUTION.md            # 纯 harness 进化：完整方法 + 全量结果
├── KEV_THREE_AXIS.md       # 三轴 kev 先验：词表 + prior/enforce 实现 + 接口
└── META_AGENT_DECISIONS.md # meta-LLM 的 14 个决断清单
```

> 已归档（不再活跃）：gridgames / ragen 线在 `archive/recipe/agent_evolver/`，A3 rule 注入臂在 `archive/rule_arm/`，规模扫描脚本在 `archive/scripts/`。复活方式 = `git mv` 回原位。

---

## 4. 复现步骤

### 4.1 前置依赖

- **机器**：4× GPU（M402 80GB）。单机三轴 = 4 卡够；16 卡 cloudml 见 §4.4。
- **agent 模型**：`/mnt/llmshared-ssd-hd/shishuqing/models/Qwen3.5-4B`（vLLM serve，冻结）。
- **kev sidecar**：`/mnt/llmshared-ssd-hd/shishuqing/kev`（代码）+ `/mnt/llmshared-ssd-hd/shishuqing/models/kev-4b`（训练产物 head.pt）。kev 的 base 模型 `Qwen3.5-4B-Base` 需 symlink 进 HF cache（`three_axis_cloudml.sh` 里的 `kev_hf_cache()` 已处理）。
- **env server 依赖**：ALFWorld 用 conda env `ee-alfworld`、WebShop 用 `ee-webshop`（含 Java + spacy），路径见 `start_three_axis_infra.sh`。
- **meta 后端**：deepseek-v4-pro（mioffice / anthropic SDK）。多机 pod 够不到 mioffice 时自动回退 pod 挂载的 `/preset-models/public/DeepSeek-V4.1-Flash` 本地 vLLM（`three_axis_cloudml.sh` 已内置）。

### 4.2 一键：单机三轴对照（推荐）

```bash
cd /mnt/llmshared-ssd-hd/shishuqing/Harness-JEV

# ① 起 infra（vLLM :8200-8203 + kev :8090 + env servers），bench = alfworld|webshop
bash start_three_axis_infra.sh alfworld

# ② 跑三臂 + held-out 探针（llm → prior → enforce 串行，逐臂 --decision-mode 不同）
setsid nohup bash three_axis_q0.sh alfworld >> /tmp/three_axis_q0.log 2>&1 &
```

`start_three_axis_infra.sh` 起的端口：

| 服务 | 端口 | 说明 |
|---|---|---|
| vLLM（agent Qwen3.5-4B） | :8200–8203 | gpu-mem 0.70（给 GPU0 上 kev 让显存） |
| kev sidecar | :8090 | 冻结读头，prior/enforce 臂都要 |
| ALFWorld held-in | :18082–18085 | `eval_in_distribution` |
| ALFWorld held-out | :18086–18089 | `eval_out_of_distribution`（探针期拉起） |
| WebShop | :18090 | 官方 full 集，12087 human goals，seed 233 |

`three_axis_q0.sh` 做三件事：① `llm` 臂（无 decision backend）→ ② `prior` 臂（`--decision-backend kev --decision-mode prior`）→ ③ `enforce` 臂（`--decision-mode enforce`），各 6 轮×64 task；最后 4 个 held-out 探针（baseline + 各臂 best config）并打印汇总。

### 4.3 手动 CLI（逐臂 / 探针）

不想用脚本时，逐臂跑：

```bash
export EVOLVER_AGENT_TEMPERATURE=0   # 确定性 rollout（三轴对照必须）

# llm 臂（无先验）
./.venv/bin/python -m recipe.agent_evolver.run alfworld --split heldin \
  --num-rounds 6 --num-tasks 64 --run-tag my_llm_heldin

# prior 臂（kev 软先验）
./.venv/bin/python -m recipe.agent_evolver.run alfworld --split heldin \
  --num-rounds 6 --num-tasks 64 --run-tag my_prior_heldin \
  --decision-backend kev --decision-mode prior --decision-base-url http://127.0.0.1:8090

# enforce 臂（kev 硬绑定 + candidate retrocheck）
./.venv/bin/python -m recipe.agent_evolver.run alfworld --split heldin \
  --num-rounds 6 --num-tasks 64 --run-tag my_enforce_heldin \
  --decision-backend kev --decision-mode enforce --decision-base-url http://127.0.0.1:8090

# held-out 探针（取某臂 best 轮 config 重放，只测不进化）
./.venv/bin/python -m recipe.agent_evolver.run alfworld --split heldout \
  --num-rounds 1 --num-tasks 64 --run-tag my_prior_heldout \
  --base-config recipe/agent_evolver/runs/evolve/my_prior_heldin/R5/config.yaml
```

WebShop 同构，只是：`webshop` 无 `--split`、held-in 用 `--start 1500`、held-out 用 `--start 0`、env 用 `--env-urls http://127.0.0.1:18090`。

### 4.4 多机（cloudml 16 卡）

cloudml/Volcano 提交一个 **2 pod × 8 GPU** 的 PyTorch job，`DOCKER_COMMAND` 填：

```bash
# BENCH=alfworld（默认）| webshop（env 或 $1）；RANK0 跑 llm→enforce、RANK1 跑 prior，共享盘 join
bash /mnt/llmshared-ssd-hd/shishuqing/Harness-JEV/three_axis_cloudml.sh
```

脚本自持 vLLM/kev/env 起停 + RANK 分派 + held-out 探针 + summary + 精确 PID teardown。meta 后端自动 detect pod 挂载的 preset 模型。提交前可用 `multi_node/verify_meta_pod.sh` 单 pod 探一下（① preset 挂载 ② vLLM dry-run ③ mioffice 可达性）。

### 4.5 读结果

每个 run 目录（`runs/evolve/<tag>/`）下：

- `comparison.json` —— 逐轮 `pass_rate` + `mean_reward`（**权威数字**，best = mean_reward 最高轮）。
- `R{n}/trajectories/{task_id}.md` —— 每 task 完整轨迹（frontmatter 带 `eval_passed`/`eval_score`）。
- `R{n}/evolve/decision_priors.md`（prior/enforce）—— kev 注入的失败机制先验。
- `R{n}/evolve/candidate_retrocheck.md`（enforce）—— intent 反事实校验。
- `learnings.md` —— meta-agent 的跨轮记忆。

> ⚠️ **别数中途日志的 win 数**：OK 任务快、FAIL 任务慢，中途数日志会虚高（一次 94% 实际 58.8%）。只看 `comparison.json`。

---

## 5. 环境变量 + CLI 参考

### 5.1 环境变量（`spec.py` 全部可覆写）

| 变量 | 默认 | 含义 |
|---|---|---|
| `EVOLVER_AGENT_MODEL` | `Qwen3.5-4B` | **agent 模型 id（切模型唯一开关，`--agent-model` 是死代码）** |
| `EVOLVER_AGENT_API_BASE` | `http://127.0.0.1:8200/v1` | agent vLLM base |
| `EVOLVER_AGENT_TEMPERATURE` | 0.4 | 生成温度（**三轴对照设 0 做确定性**） |
| `EVOLVER_AGENT_MAX_TOKENS` | 4096 | thinking on 需要大预算 |
| `EVOLVER_META_MODEL` / `_META_API_BASE` | deepseek-v4-pro | meta-agent 后端 |

### 5.2 CLI 关键参数（`python -m recipe.agent_evolver.run --help`）

| 参数 | 说明 |
|---|---|
| `{alfworld,webshop}` | benchmark（positional） |
| `--split heldin\|heldout` | ALF 专属；WS 忽略（用 `--start` 切） |
| `--num-rounds N` | 总轮数（R0..R{N-1}；R0=裸基线） |
| `--num-tasks N` / `--start K` / `--seed S` | 采样：数量 / 起始 task_id / seed |
| `--decision-backend {llm,kev}` | `llm`=关；`kev`=注入 System-1 先验 |
| `--decision-base-url URL` | kev sidecar 地址（默认 :8090） |
| `--decision-mode {prior,enforce}` | 软先验 / 硬绑定 |
| `--base-config YAML` | 探针用：从外部 config 起跑（不进化） |
| `--agent-api-bases` / `--env-urls` | 多卡 vLLM / env server 分片端点 |
| `--run-tag TAG` | 输出目录名 |

---

## 6. 坑

- **`--agent-model` 是死代码**：切模型只认 `EVOLVER_AGENT_MODEL`，且 vLLM 的 `--served-model-name` 必须与之相等。
- **temp 必须是 0**：三轴对照逐 seed 比较时，vLLM bf16/重启噪声会污染 ±2–4pt；只有 temp=0 贪心才在同 run 内可比（跨 run 绝对值仍要打折）。
- **ALF env 是 1 worker = 1 env server 的 stateful shard**（`env_urls[i % len]`）：**别让两个 run 同时占同一 held-in/held-out 端口**，否则并发打同一 server → `RemoteProtocolError`（STEP-CONN-ERROR 记 task 失败）。
- **enforce 的硬 fail**：`--decision-mode enforce` 且后端挂 → loud abort（不静默回退），这是设计（防止硬 gate 假装通过）。
- **meta 后端**：mioffice fork 无 `temperature` 参数、`max_tokens ≥ 4096`（否则 thinking 吃光预算）；`LITELLM_LOCAL_MODEL_COST_MAP=True` 必须在 `import litellm` 前设（否则每进程 ~45s）。
- **infra 会被清**：机器重启 / supervisord 周期清掉 vLLM 和 env server，报 `httpx.ConnectError` 先重跑 `start_three_axis_infra.sh`。

---

## 7. 相关文档

| 文档 | 内容 |
|---|---|
| [EVOLUTION.md](EVOLUTION.md) | 纯 harness 进化：数据流图、可进化面、模型矩阵、WebShop 迁移、gridgames decoding |
| [KEV_THREE_AXIS.md](KEV_THREE_AXIS.md) | 三轴词表定义、prior/enforce 实现细节、改动文件清单 |
| [META_AGENT_DECISIONS.md](META_AGENT_DECISIONS.md) | meta-LLM 的 14 个决断 + JEV 可替代性分析 |
| [../../RESULTS.md](../../RESULTS.md) | 全结果汇总（ALF/WS/FL/SK + 规模矩阵 + 三轴 + 系统-1 参照系） |
| [../../REPORT_visibility.md](../../REPORT_visibility.md) | visibility 报告（S2 基座 + A1/A2 + B1 + 三轴 + P1-P9 判定 + 机制讨论） |
