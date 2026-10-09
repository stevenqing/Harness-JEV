#!/usr/bin/env bash
# Bring up the WebShop three-axis (llm vs prior vs enforce) infra, in order.
# Everything is setsid+nohup so it survives session exit. Idempotent: skips
# anything already up.
#
#   vLLM x4            :8200-8203  (Qwen3.5-4B, GPU 0..3, gpu-mem 0.70)
#   kev serve x1       :8090       (4B-Base+LoRA+head, GPU 0)  <-- decision backend
#   WebShop env x1     :18090      (official full set, 12087 human goals, seed 233)
#
# WebShop 与 ALFWorld 不同：env server 只有【一个】(ThreadingHTTPServer 线程安全、
# 每次 /create 独立 uuid 实例可吸收 N 并发 worker)，held-in/held-out 靠 run.py 的
# --start 切（train 1500.. / test 0..），所以无需 held-out env。
#
# KEY: vLLM runs at 0.70 (not 0.85) so kev (~19GiB) can co-reside on GPU 0.
#
# After infra is up, launch the experiment:
#   setsid nohup bash three_axis_webshop_q0.sh >> /tmp/three_axis_webshop_q0.log 2>&1 &
#
# Usage:  bash start_webshop_three_axis_infra.sh
set -uo pipefail

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
MODEL=/mnt/llmshared-ssd-hd/shishuqing/models/Qwen3.5-4B
VLLM=/mnt/llmshared-ssd-hd/shishuqing/.venv/bin/vllm
KEV_REPO=/mnt/llmshared-ssd-hd/shishuqing/kev
KEV_RUN=/mnt/llmshared-ssd-hd/shishuqing/models/kev-4b
WS_PY=/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-webshop/bin/python
WS_JAVA_HOME=/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-webshop/lib/jvm
WS_DATA=/mnt/llmshared-ssd-hd/chenjinyuan/memory-agent/data/webshop_official_64fa2a5
WS_ROOT="$REPO/benchmarks/webshop/official_webshop"

vllm_up() { curl -s --max-time 3 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
env_up()  { curl -s --max-time 3 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }
kev_up()  { curl -s --max-time 3 "http://127.0.0.1:8090/v1/models" >/dev/null 2>&1; }

log() { echo "[wsinfra $(date +%H:%M:%S)] $*"; }

# ── Pre-flight path checks ───────────────────────────────────────────────────
ok=1
for p in "$MODEL" "$VLLM" "$KEV_REPO/.venv/bin/python" "$KEV_RUN/head.pt" \
         "$WS_PY" "$WS_JAVA_HOME" "$WS_DATA/items_shuffle.json" "$WS_DATA/items_ins_v2.json" \
         "$WS_ROOT/web_agent_site"; do
  [ -e "$p" ] || { log "MISSING: $p"; ok=0; }
done
[ "$ok" = 0 ] && { log "aborting — missing paths above"; exit 1; }
log "path checks OK"

# ── 1. vLLM x4 (one per GPU, gpu-mem 0.70) ───────────────────────────────────
for i in 0 1 2 3; do
  port=$((8200 + i))
  if vllm_up "$port"; then log "vLLM :$port already up"; continue; fi
  log "launching vLLM :$port on GPU $i"
  CUDA_VISIBLE_DEVICES=$i setsid nohup "$VLLM" serve "$MODEL" \
    --host 127.0.0.1 --port "$port" --served-model-name Qwen3.5-4B \
    --gpu-memory-utilization 0.70 --max-model-len 32768 --max-num-seqs 256 \
    --tensor-parallel-size 1 --dtype bfloat16 --no-enable-log-requests --max-logprobs 40 \
    > "/tmp/vllm_$port.log" 2>&1 &
done

# ── 2. kev serve x1 (GPU 0, decision backend for prior/enforce arms) ─────────
if kev_up; then
  log "kev :8090 already up"
else
  log "launching kev serve :8090 on GPU 0"
  (
    cd "$KEV_REPO" || exit 1
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 KEV_PREFIX_CACHE=0 \
      CUDA_VISIBLE_DEVICES=0 setsid nohup .venv/bin/python -m kev.serve \
      --run "$KEV_RUN" --port 8090 > /tmp/kev_serve_8090.log 2>&1 &
  )
fi

# ── 3. WebShop env server (single :18090, official full set) ─────────────────
if env_up 18090; then
  log "WebShop :18090 already up"
else
  log "launching WebShop :18090 (official full set, seed 233 — loads ~1-2 min)"
  JAVA_HOME="$WS_JAVA_HOME" PATH="$WS_JAVA_HOME/bin:$PATH" \
    WEBSHOP_ROOT="$WS_ROOT" \
    WEBSHOP_NUM_PRODUCTS=full WEBSHOP_HUMAN_GOALS=1 WEBSHOP_SEED=233 \
    WEBSHOP_FILE_PATH="$WS_DATA/items_shuffle.json" \
    WEBSHOP_ATTR_PATH="$WS_DATA/items_ins_v2.json" \
    setsid nohup "$WS_PY" "$REPO/benchmarks/webshop/env_server.py" --host 127.0.0.1 --port 18090 \
    > /tmp/webshop_env_18090.log 2>&1 &
fi

# ── 4. Wait + health report ──────────────────────────────────────────────────
log "waiting for services to come up (vLLM ~2-4 min, WebShop env ~1-2 min) ..."
for i in $(seq 1 72); do
  v=0; for p in 8200 8201 8202 8203; do vllm_up "$p" && v=$((v+1)); done
  k=0; kev_up && k=1
  e=0; env_up 18090 && e=1
  log "vLLM $v/4  kev $k/1  webshop_env $e/1"
  [ "$v" = 4 ] && [ "$k" = 1 ] && [ "$e" = 1 ] && { log "ALL INFRA UP"; break; }
  sleep 5
done

# ── Final status (authoritative) ─────────────────────────────────────────────
log "=== FINAL INFRA STATUS ==="
for p in 8200 8201 8202 8203; do vllm_up "$p" && log "vLLM :$p UP" || log "vLLM :$p DOWN"; done
kev_up && log "kev :8090 UP" || log "kev :8090 DOWN (see /tmp/kev_serve_8090.log)"
if env_up 18090; then
  log "WebShop :18090 UP ($(curl -s --max-time 3 http://127.0.0.1:18090/health))"
else
  log "WebShop :18090 DOWN (see /tmp/webshop_env_18090.log)"
fi
log "done — if everything is UP, launch: setsid nohup bash three_axis_webshop_q0.sh >> /tmp/three_axis_webshop_q0.log 2>&1 &"
