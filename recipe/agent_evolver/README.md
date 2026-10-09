# Harness-JEV agent_evolver —— 纯 harness 进化（模型规模扫描）

> 只进化 **harness**（system prompt / guidance / loop 参数），模型权重永不训练。
> 本 README 讲清「做了什么、怎么复现」。方法与全量结果见 [EVOLUTION.md](EVOLUTION.md)。

---

## 0. 一句话

用同一套文本协议进化器（`run.py`）换 agent 底座跑 **Qwen3.5-{4B, 2B, 0.8B} × {ALFWorld, WebShop}** 各 6 轮，回答一个问题：**纯 harness 进化对弱模型的增益随规模如何变化**。结论：绝对增益随规模坍缩，但相对增益（百分比）反而随规模下降而上升；二值 reward 任务存在硬能力地板（0.8B 进化到 0.078 且零泛化），连续 reward 任务即便 0.8B 也能被抬 24 倍。

---

## 1. 做了什么（结果）

协议（三档完全一致）：

- **ALFWorld**：held-in（`valid_seen`，140 可解局）进化 6 轮 → held-out（`valid_unseen`，134）只测一次（`--base-config` 指进化最佳轮）。
- **WebShop**：官方 train（`--start 1500`，人类目标 1500–12086 切片）进化 6 轮 → test（`--start 0`，目标 0–499 切片）只测一次。

**总表**（pass 率；WS 括号内 mean_reward）：

| 模型 | ALF held-in 基线→best | ALF held-out | WS train 基线→best | WS test |
|---|---|---|---|---|
| **4B** | 0.484 → **0.703** (R2) | 0.750 | 0.188(0.472) → 0.188(**0.556**, R2) | 0.219(0.559) / 128-task 0.242(0.568) |
| **2B** | 0.109 → **0.250** (R3) | 0.219 | 0.141(0.550) → 0.250(**0.646**, R4) | 0.297(0.656) |
| **0.8B** | 0.000 → **0.078** (R3) | 0.000 | 0.000(0.016) → 0.125(**0.394**, R4) | 0.109(0.343) |

**三条结论**：

1. **绝对分数随规模单调坍缩**：ALF best 0.703 / 0.250 / 0.078，基线 0.484 / 0.109 / 0.000。
2. **二值 vs 连续 reward 决定「能力地板」是否致命**：ALFWorld（二值 success）0.8B 三轮全 0、第 3 轮才 0.078 且 held-out 归零；WebShop（连续 reward）0.8B 基线 mean 0.016 → best 0.394（×24）、test 0.343 保留 87%。
3. **相对增益随规模下降而升、绝对增益反之**：WS mean Δ 4B +8.4 / 2B +9.5 / 0.8B +37.8pt。

一句话：**进化补的是「知道怎么走」，补不了「走不动」**。

---

## 2. 原理（怎么进化的）

每轮循环（`run.py:283` 起的 `for round_idx in range(num_rounds)`）：

```
R0 裸基线 ──rollout 64 task──▶ 写 trajectories ──gate──▶ meta_agent.evolve() ──▶ R1 config
                                                                                    │
   R1 rollout ──gate──▶ evolve ──▶ R2 ...   （gate: mean_reward ≥ best − 0.02 才 accept，否则回退 best）
```

- **进化载体是 `HarnessConfig` YAML**：`SystemPromptProcessor`（裸 Qwen 默认 prompt，从未被改）+ `TextEvolverSpecProcessor`（`guidance` 文本 + `history_n`/`loop_breaker_k`/`history_mode` 等 loop 标量）。
- **meta-agent = 强模型**（默认 `anthropic/volcengine_maas/deepseek-v4-pro`）读轨迹、写 guidance；它发现的「任务评分语法」（ALFWorld 动作词表、WebShop 点击每个属性选项）是增益来源。
- **探针**：进化完取 `comparison.json` 里 mean_reward 最高的那轮 config，`--base-config` 指过去、在 held-out/test 上只跑 R0（`--num-rounds 1`）看是否真泛化。

---

## 3. 目录结构

```
recipe/agent_evolver/
├── run.py            # 进化器主循环（rollout + gate + evolve + journal）
├── spec.py           # HarnessSpec 数据类 + 环境变量默认值（AGENT_MODEL 在这里读）
├── gate.py           # 回归门（accept/revert）
├── trajectories.py   # per-task .md 轨迹写入
├── native/           # HarnessConfig 构造 + processor + prompt builder
├── skills/           # meta-agent 的 playbook skill（alfworld / webshop）
├── runs/evolve/<tag>/   # 每次 run 的产物：R{n}/config.yaml + trajectories/ + comparison.json
├── EVOLUTION.md      # 方法与全量结果
└── META_AGENT_DECISIONS.md  # meta 里 LLM 的决断清单
```

---

## 4. 复现步骤

### 4.1 前置

- 机器：4× GPU（M402 80GB），vLLM 装在 `/mnt/llmshared-ssd-hd/shishuqing/.venv`。
- agent 模型（同一 `Qwen3_5ForConditionalGeneration` 家族）：
  - 4B  `/mnt/llmshared-ssd-hd/shishuqing/models/Qwen3.5-4B`
  - 2B  `/mnt/llmshared-ssd-hd/hanguangzeng/models/qwen3.5-2b`
  - 0.8B `/mnt/llmshared-ssd-hd/hanguangzeng/models/qwen3.5-0.8b`
- env server 依赖：ALFWorld 用 `ee-alfworld` 环境、WebShop 用 `ee-webshop`（路径见 `benchmarks/start_jev_infra.sh`）。
- meta 后端（deepseek-v4-pro）可用（mioffice / anthropic SDK）。

### 4.2 起 infra（vLLM + env server）

```bash
bash benchmarks/start_jev_infra.sh
```

默认起 **4B** vLLM（:8200–8203）+ ALF held-in（:18082–18085）/ held-out（:18086–18089）+ WebShop（:18090）。

> ⚠️ 该脚本默认 serve 4B。跑 2B/0.8B 前先停掉 4B、换模型重 serve（见 4.3）。

### 4.3 切 agent 模型（关键）

`run.py` 里 agent 模型**只看环境变量 `EVOLVER_AGENT_MODEL`**（`--agent-model` 参数是死代码，只影响报告行）。所以切模型 = 两件事：

```bash
# ① 停掉当前 vLLM（逐个 kill PID，别用 pkill -f，会自匹配）
ps aux | grep 'vllm serve' | grep -v grep | awk '{print $2}' | while read p; do kill "$p"; done

# ② 起目标模型的 vLLM（served-model-name 必须 = EVOLVER_AGENT_MODEL 的值）
VLLM=/mnt/llmshared-ssd-hd/shishuqing/.venv/bin/vllm
MODEL=/mnt/llmshared-ssd-hd/hanguangzeng/models/qwen3.5-2b   # 换 0.8B 改这里
for i in 0 1 2 3; do
  port=$((8200 + i))
  CUDA_VISIBLE_DEVICES=$i setsid nohup "$VLLM" serve "$MODEL" \
    --host 127.0.0.1 --port "$port" --served-model-name Qwen3.5-2B \
    --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 256 \
    --tensor-parallel-size 1 --dtype bfloat16 --no-enable-log-requests --max-logprobs 40 \
    > "/tmp/vllm_2b_$port.log" 2>&1 &
done
# 就绪校验
curl -s http://127.0.0.1:8200/v1/models   # 应返回 {"id":"Qwen3.5-2B", ...}
```

### 4.4 跑进化（6 轮）

```bash
cd /mnt/llmshared-ssd-hd/shishuqing/Harness-JEV

# ALFWorld held-in 进化
EVOLVER_AGENT_MODEL=Qwen3.5-2B ./.venv/bin/python -m recipe.agent_evolver.run \
  alfworld --split heldin --num-rounds 6 --num-tasks 64 --run-tag alfheldin-2b-v1

# WebShop train 进化
EVOLVER_AGENT_MODEL=Qwen3.5-2B ./.venv/bin/python -m recipe.agent_evolver.run \
  webshop --num-rounds 6 --num-tasks 64 --start 1500 --run-tag ws-train-2b-v1
```

- `--num-rounds 6` = R0（裸基线）+ R1..R5（5 次 evolve）。
- ALF seed 默认 1234、WS seed 0；64 task = ALF game_idx 0–63 / WS goals 1500–1563。

### 4.5 跑泛化探针（只测，不进化）

取进化 run 里 mean_reward 最高轮的 config，`--base-config` 指过去、`--num-rounds 1`：

```bash
# 找 best 轮（mean_reward 最高）
python3 -c "import json;d=json.load(open('recipe/agent_evolver/runs/evolve/alfheldin-2b-v1/comparison.json'));print(max(d['rounds'],key=lambda r:r['mean_reward'])['round'])"

# ALFWorld held-out 探针（假设 best=R3）
EVOLVER_AGENT_MODEL=Qwen3.5-2B ./.venv/bin/python -m recipe.agent_evolver.run \
  alfworld --split heldout --num-rounds 1 --num-tasks 64 \
  --base-config recipe/agent_evolver/runs/evolve/alfheldin-2b-v1/R3/config.yaml \
  --run-tag alfheldout-2b-v1

# WebShop test 探针（假设 best=R4）
EVOLVER_AGENT_MODEL=Qwen3.5-2B ./.venv/bin/python -m recipe.agent_evolver.run \
  webshop --num-rounds 1 --num-tasks 64 --start 0 \
  --base-config recipe/agent_evolver/runs/evolve/ws-train-2b-v1/R4/config.yaml \
  --run-tag ws-test-2b-v1
```

### 4.6 读结果

每个 run 目录下：

- `comparison.json` —— 逐轮 `pass_rate` + `mean_reward`（权威数字）。
- `R{n}/trajectories/{task_id}.md` —— 每 task 的完整轨迹（frontmatter 带 `eval_passed`/`eval_score`）。
- `learnings.md` —— meta-agent 的跨轮记忆 / 归因。

### 4.7 一键驱动脚本（可选）

上面 4 阶段（ALF 进化→探针→WS 进化→探针）可用一个自编排脚本串起来，自动选 best 轮：

```bash
#!/usr/bin/env bash
# usage: run_matrix.sh <served-model-name> <label>
MODEL_NAME="$1"; LABEL="$2"
ROOT=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$ROOT/.venv/bin/python"; RUNS="$ROOT/recipe/agent_evolver/runs/evolve"
export EVOLVER_AGENT_MODEL="$MODEL_NAME"; cd "$ROOT"

best_round() { python3 - "$RUNS/$1/comparison.json" <<'PY'
import sys,json; d=json.load(open(sys.argv[1])); print(max(d["rounds"],key=lambda r:r["mean_reward"])["round"])
PY
}

"$PY" -m recipe.agent_evolver.run alfworld --split heldin --num-rounds 6 --num-tasks 64 --run-tag "alfheldin-$LABEL-v1"
B=$(best_round "alfheldin-$LABEL-v1")
"$PY" -m recipe.agent_evolver.run alfworld --split heldout --num-rounds 1 --num-tasks 64 --base-config "$RUNS/alfheldin-$LABEL-v1/R$B/config.yaml" --run-tag "alfheldout-$LABEL-v1"

"$PY" -m recipe.agent_evolver.run webshop --num-rounds 6 --num-tasks 64 --start 1500 --run-tag "ws-train-$LABEL-v1"
B=$(best_round "ws-train-$LABEL-v1")
"$PY" -m recipe.agent_evolver.run webshop --num-rounds 1 --num-tasks 64 --start 0 --base-config "$RUNS/ws-train-$LABEL-v1/R$B/config.yaml" --run-tag "ws-test-$LABEL-v1"
```

---

## 5. 环境变量（spec.py 全部可覆写）

| 变量 | 默认 | 含义 |
|---|---|---|
| `EVOLVER_AGENT_MODEL` | `Qwen3.5-4B` | **agent 模型 id（切模型唯一开关）** |
| `EVOLVER_AGENT_API_BASE` | `http://127.0.0.1:8200/v1` | agent vLLM base |
| `EVOLVER_AGENT_TEMPERATURE` / `_TOP_P` / `_MAX_TOKENS` | 0.4 / 0.8 / 1024 | 生成参数 |
| `EVOLVER_HISTORY_LENGTH` | 2 | 注入 history 的 (obs,action) 对数 |
| `EVOLVER_ALF_MAX_STEPS` / `_WS_MAX_STEPS` | 50 / 15 | 每 task 步数上限 |
| `EVOLVER_META_MODEL` / `_META_API_BASE` | deepseek-v4-pro | meta-agent 后端 |

CLI 参数（`python -m recipe.agent_evolver.run --help`）覆盖这些默认；`--base-config` 用于探针，`--split heldin/heldout` 仅 ALF，`--start` 切 WS train/test。

---

## 6. 坑

- **`--agent-model` 是死代码**：切模型只认 `EVOLVER_AGENT_MODEL`，否则跑的还是 4B。
- **跨 run 有 ±2–4pt 的 vLLM bf16/重启噪声**：探针必须和「同模型的进化 run」背靠背跑才可比；跨 run 绝对数值只在同 run 内可做逐轮 gate 比较。
- **meta 后端**：`AnthropicProvider` 要 `extended_thinking=False`、`max_tokens ≥ 4096`（mioffice fork 无 `temperature`）。
- **LiteLLM 延迟**：`LITELLM_LOCAL_MODEL_COST_MAP=True` 必须在 `import litellm` 前设，否则每进程 ~45s。
- **infra 会被清**：机器重启 / supervisord 周期会清掉 vLLM 和 WebShop，报 `httpx.ConnectError` 时先重跑 `start_jev_infra.sh`。
- **全失败轨迹**（如 0.8B ALF 前三轮）里没有成功信号，meta 收敛更慢，属预期。
