#!/usr/bin/env bash
# Bring up the three-axis (llm vs prior vs enforce) experiment infra, in order.
# Everything is setsid+nohup so it survives session exit. Idempotent: skips
# anything already up.
#
#   vLLM x4            :8200-8203  (Qwen3.5-4B, GPU 0..3, gpu-mem 0.70)
#   kev serve x1       :8090       (4B-Base+LoRA+head, GPU 0)  <-- decision backend
#
#   alfworld:
#     ALFWorld held-in  x4 :18082-18085 (eval_in_distribution)
#     ALFWorld held-out x4 :18086-18089 (eval_out_of_distribution)
#   webshop:
#     WebShop env x1     :18090  (official full set, 12087 human goals, seed 233)
#     — WebShop 只有一个 env server（ThreadingHTTPServer 线程安全、每次 /create 独立
#       uuid 实例可吸收 N 并发 worker），held-in/held-out 靠 run.py 的 --start 切，无需 held-out env。
#
# KEY: vLLM runs at 0.70 (not the default 0.85) so the kev serve (~19GiB) can
# co-reside on GPU 0. 0.70 is ample for the rollout workload (max ~1024 tok/req).
# Raise back to 0.85 only when kev is NOT co-resident on the same cards.
#
# After infra is up, launch the experiment:
#   setsid nohup bash three_axis_q0.sh <alfworld|webshop> >> /tmp/three_axis_q0.log 2>&1 &
#
# Usage:  bash start_three_axis_infra.sh [alfworld|webshop]
set -uo pipefail

BENCH="${1:-${BENCH:-alfworld}}"
case "$BENCH" in
  alfworld|webshop) ;;
  *) echo "usage: $0 [alfworld|webshop]" >&2; exit 2 ;;
esac

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
MODEL=/mnt/llmshared-ssd-hd/shishuqing/models/Qwen3.5-4B
VLLM=/mnt/llmshared-ssd-hd/shishuqing/.venv/bin/vllm
KEV_REPO=/mnt/llmshared-ssd-hd/shishuqing/kev
KEV_RUN=/mnt/llmshared-ssd-hd/shishuqing/models/kev-4b
ALF_PY=/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-alfworld/bin/python
ALF_DATA=/mnt/llmshared-ssd-hd/cty/data/alfworld
WS_PY=/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-webshop/bin/python
WS_JAVA_HOME=/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-webshop/lib/jvm
WS_DATA=/mnt/llmshared-ssd-hd/chenjinyuan/memory-agent/data/webshop_official_64fa2a5
WS_ROOT="$REPO/benchmarks/webshop/official_webshop"

vllm_up() { curl -s --max-time 3 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
env_up()  { curl -s --max-time 3 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }
kev_up()  { curl -s --max-time 3 "http://127.0.0.1:8090/v1/models" >/dev/null 2>&1; }

log() { echo "[infra/$BENCH $(date +%H:%M:%S)] $*"; }

# ── Pre-flight path checks ───────────────────────────────────────────────────
ok=1
for p in "$MODEL" "$VLLM" "$KEV_REPO/.venv/bin/python" "$KEV_RUN/head.pt"; do
  [ -e "$p" ] || { log "MISSING: $p"; ok=0; }
done
if [ "$BENCH" = alfworld ]; then
  for p in "$ALF_PY" "$ALF_DATA"; do
    [ -e "$p" ] || { log "MISSING: $p"; ok=0; }
  done
else
  for p in "$WS_PY" "$WS_JAVA_HOME" "$WS_DATA/items_shuffle.json" "$WS_DATA/items_ins_v2.json" "$WS_ROOT/web_agent_site"; do
    [ -e "$p" ] || { log "MISSING: $p"; ok=0; }
  done
fi
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

# ── 3. benchmark env server(s) ────────────────────────────────────────────────
if [ "$BENCH" = alfworld ]; then
  for p in 18082 18083 18084 18085; do
    if env_up "$p"; then log "ALF held-in :$p already up"; continue; fi
    ALFWORLD_DATA="$ALF_DATA" ALFWORLD_SPLIT=eval_in_distribution \
      setsid nohup "$ALF_PY" "$REPO/benchmarks/alfworld/env_server.py" --port "$p" \
      > "/tmp/alfworld_heldin_$p.log" 2>&1 &
    log "ALF held-in :$p launching"
  done
  for p in 18086 18087 18088 18089; do
    if env_up "$p"; then log "ALF held-out :$p already up"; continue; fi
    ALFWORLD_DATA="$ALF_DATA" ALFWORLD_SPLIT=eval_out_of_distribution \
      setsid nohup "$ALF_PY" "$REPO/benchmarks/alfworld/env_server.py" --port "$p" \
      > "/tmp/alfworld_heldout_$p.log" 2>&1 &
    log "ALF held-out :$p launching"
  done
else
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
fi

# ── 4. Wait + health report ──────────────────────────────────────────────────
if [ "$BENCH" = alfworld ]; then
  ENV_PORTS=(18082 18083 18084 18085 18086 18087 18088 18089)
  ENV_NEED=8
  ENV_LABEL="env"
else
  ENV_PORTS=(18090)
  ENV_NEED=1
  ENV_LABEL="webshop_env"
fi
log "waiting for services to come up (up to ~4 min for vLLM) ..."
for i in $(seq 1 48); do
  v=0; for p in 8200 8201 8202 8203; do vllm_up "$p" && v=$((v+1)); done
  k=0; kev_up && k=1
  e=0; for p in "${ENV_PORTS[@]}"; do env_up "$p" && e=$((e+1)); done
  log "vLLM $v/4  kev $k/1  $ENV_LABEL $e/$ENV_NEED"
  [ "$v" = 4 ] && [ "$k" = 1 ] && [ "$e" = "$ENV_NEED" ] && { log "ALL INFRA UP"; break; }
  sleep 5
done

# ── Final status (authoritative) ─────────────────────────────────────────────
log "=== FINAL INFRA STATUS ==="
for p in 8200 8201 8202 8203; do vllm_up "$p" && log "vLLM :$p UP" || log "vLLM :$p DOWN"; done
kev_up && log "kev :8090 UP" || log "kev :8090 DOWN (see /tmp/kev_serve_8090.log)"
if [ "$BENCH" = alfworld ]; then
  for p in 18082 18083 18084 18085; do env_up "$p" && log "ALF held-in :$p UP" || log "ALF held-in :$p DOWN"; done
  for p in 18086 18087 18088 18089; do env_up "$p" && log "ALF held-out :$p UP" || log "ALF held-out :$p DOWN"; done
else
  if env_up 18090; then
    log "WebShop :18090 UP ($(curl -s --max-time 3 http://127.0.0.1:18090/health))"
  else
    log "WebShop :18090 DOWN (see /tmp/webshop_env_18090.log)"
  fi
fi
log "done — if everything is UP, launch: setsid nohup bash three_axis_q0.sh $BENCH >> /tmp/three_axis_q0.log 2>&1 &"
