#!/usr/bin/env bash
# Bring up everything agent_evolver needs: vLLM (Qwen3.5-4B :8200) + both env servers.
# Usage:  bash benchmarks/start_all.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM="/mnt/llmshared-ssd-hd/shishuqing/.venv/bin/vllm"
MODEL="$REPO/models/Qwen3.5-4B"

# env servers (18081/18082)
bash "$REPO/benchmarks/start_env_servers.sh"

# vLLM
if ! curl -s --max-time 3 http://127.0.0.1:8200/v1/models >/dev/null 2>&1; then
  echo "[start] vLLM Qwen3.5-4B :8200"
  nohup "$VLLM" serve "$MODEL" \
    --host 127.0.0.1 --port 8200 \
    --served-model-name Qwen3.5-4B \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --no-enable-log-requests \
    --max-logprobs 40 \
    > /tmp/vllm_qwen35_4b.log 2>&1 &
else
  echo "[start] vLLM :8200 already up"
fi

echo "[start] waiting for vLLM ..."
for i in $(seq 1 60); do
  if curl -s --max-time 3 http://127.0.0.1:8200/v1/models >/dev/null 2>&1; then
    echo "[start] ready: $(curl -s --max-time 3 http://127.0.0.1:8200/v1/models | head -c 60)"
    exit 0
  fi
  sleep 5
done
echo "[start] TIMEOUT waiting for vLLM; check /tmp/vllm_qwen35_4b.log"
exit 1
