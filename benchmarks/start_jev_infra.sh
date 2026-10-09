#!/usr/bin/env bash
# Bring up the Harness-JEV agent_evolver infra (all setsid+nohup so they survive
# session exit):
#   vLLM x4            :8200-8203  (Qwen3.5-4B, GPU 0..3)
#   ALFWorld held-in  x4 :18082-18085 (eval_in_distribution, game_idx 0-63)
#   ALFWorld held-out x4 :18086-18089 (eval_out_of_distribution, game_idx 0-63)
#   WebShop official  x1 :18090      (118万 products, 12087 human goals, seed 233)
#
# Usage:  bash benchmarks/start_jev_infra.sh
set -uo pipefail

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
MODEL=/mnt/llmshared-ssd-hd/shishuqing/models/Qwen3.5-4B
VLLM=/mnt/llmshared-ssd-hd/shishuqing/.venv/bin/vllm
ALF_PY=/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-alfworld/bin/python
WS_PY=/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-webshop/bin/python
JAVA_HOME=/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-webshop/lib/jvm
ALF_DATA=/mnt/llmshared-ssd-hd/cty/data/alfworld
WS_DATA=/mnt/llmshared-ssd-hd/chenjinyuan/memory-agent/data/webshop_official_64fa2a5

vllm_up() { curl -s --max-time 3 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
env_up()  { curl -s --max-time 3 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }

# ── vLLM (one per GPU) ───────────────────────────────────────────────────────
for i in 0 1 2 3; do
  port=$((8200 + i))
  if vllm_up "$port"; then echo "[infra] vLLM :$port already up"; continue; fi
  echo "[infra] launching vLLM :$port on GPU $i"
  CUDA_VISIBLE_DEVICES=$i setsid nohup "$VLLM" serve "$MODEL" \
    --host 127.0.0.1 --port "$port" --served-model-name Qwen3.5-4B \
    --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 256 \
    --tensor-parallel-size 1 --dtype bfloat16 --no-enable-log-requests --max-logprobs 40 \
    > "/tmp/vllm_$port.log" 2>&1 &
done

# ── ALFWorld env servers ─────────────────────────────────────────────────────
for p in 18082 18083 18084 18085; do
  if env_up "$p"; then echo "[infra] ALF held-in :$p already up"; continue; fi
  ALFWORLD_DATA="$ALF_DATA" ALFWORLD_SPLIT=eval_in_distribution \
    setsid nohup "$ALF_PY" "$REPO/benchmarks/alfworld/env_server.py" --port "$p" \
    > "/tmp/alfworld_heldin_$p.log" 2>&1 &
done
for p in 18086 18087 18088 18089; do
  if env_up "$p"; then echo "[infra] ALF held-out :$p already up"; continue; fi
  ALFWORLD_DATA="$ALF_DATA" ALFWORLD_SPLIT=eval_out_of_distribution \
    setsid nohup "$ALF_PY" "$REPO/benchmarks/alfworld/env_server.py" --port "$p" \
    > "/tmp/alfworld_heldout_$p.log" 2>&1 &
done

# ── WebShop env server (official full set) ───────────────────────────────────
if env_up 18090; then
  echo "[infra] WebShop :18090 already up"
else
  echo "[infra] launching WebShop :18090 (official full set, seed 233)"
  JAVA_HOME="$JAVA_HOME" PATH="$JAVA_HOME/bin:$PATH" \
    WEBSHOP_ROOT="$REPO/benchmarks/webshop/official_webshop" \
    WEBSHOP_NUM_PRODUCTS=full WEBSHOP_HUMAN_GOALS=1 WEBSHOP_SEED=233 \
    WEBSHOP_FILE_PATH="$WS_DATA/items_shuffle.json" \
    WEBSHOP_ATTR_PATH="$WS_DATA/items_ins_v2.json" \
    setsid nohup "$WS_PY" "$REPO/benchmarks/webshop/env_server.py" --port 18090 \
    > /tmp/webshop_env_server_18090.log 2>&1 &
fi

echo "[infra] all launches issued"
