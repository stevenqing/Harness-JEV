#!/usr/bin/env bash
# Three-axis (llm vs prior vs enforce) scale sweep on small agent models.
#
#   Qwen3.5-2B  -> ALFWorld (three-axis) -> WebShop (three-axis)
#   Qwen3.5-0.8B -> ALFWorld (three-axis) -> WebShop (three-axis)
#
# For each model it stops the previous agent vLLM, serves the target model on
# :8200-8203 (served-model-name == model name), runs both benchmark cells, then
# switches to the next size. kev :8090 (the frozen 4B-Base+LoRA reviewer) stays
# UP throughout — it is the *decision backend* (prior/enforce arms), not the
# agent under test, so it does NOT change with agent size.
#
# S1 decoding: thinking on + max_tokens 4096 + agent temp=0 (deterministic).
#
# Launch detached so it survives session exit:
#   setsid nohup bash three_axis_scale_2b_0.8b.sh >> /tmp/three_axis_scale.log 2>&1 &
set -uo pipefail

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$REPO/.venv/bin/python"
VLLM=/mnt/llmshared-ssd-hd/shishuqing/.venv/bin/vllm
RUNS="$REPO/recipe/agent_evolver/runs/evolve"
# 64 agent workers = K=16 per GPU (num_tasks=64 caps useful concurrency; the
# decode benchmark shows near-linear batching to K=32). 4 vLLM x16 = 64 bases.
APIS=""
for _g in 0 1 2 3; do for _r in $(seq 1 16); do APIS="${APIS}http://127.0.0.1:820${_g}/v1,"; done; done
APIS="${APIS%,}"
CONC=64
ALF_HELDIN_ENV=""
for _p in $(seq 18100 18163); do ALF_HELDIN_ENV="${ALF_HELDIN_ENV}http://127.0.0.1:${_p},"; done
ALF_HELDIN_ENV="${ALF_HELDIN_ENV%,}"
ALF_HELDOUT_ENV=""
for _p in $(seq 18200 18263); do ALF_HELDOUT_ENV="${ALF_HELDOUT_ENV}http://127.0.0.1:${_p},"; done
ALF_HELDOUT_ENV="${ALF_HELDOUT_ENV%,}"
KEV_URL="http://127.0.0.1:8090"

ALF_PY=/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-alfworld/bin/python
ALF_DATA=/mnt/llmshared-ssd-hd/cty/data/alfworld
WS_PY=/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-webshop/bin/python
WS_JAVA_HOME=/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-webshop/lib/jvm
WS_DATA=/mnt/llmshared-ssd-hd/chenjinyuan/memory-agent/data/webshop_official_64fa2a5
WS_ROOT="$REPO/benchmarks/webshop/official_webshop"
WS_ENV_URL="http://127.0.0.1:18090"

export EVOLVER_AGENT_TEMPERATURE=0

log() { echo "[scale $(date +%H:%M:%S)] $*"; }

vllm_up() { curl -s --max-time 3 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
env_up()  { curl -s --max-time 3 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }
kev_up()  { curl -s --max-time 3 "$KEV_URL/" >/dev/null 2>&1; }

# ── agent vLLM lifecycle ──────────────────────────────────────────────────────
stop_agent_vllm() {
  log "stopping agent vLLM (any size) on :8200-8203"
  local pids
  pids=$(pgrep -f "vllm serve" 2>/dev/null || true)
  if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null; fi
  sleep 8
  pids=$(pgrep -f "VLLM::EngineCore" 2>/dev/null || true)
  if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null; fi
  sleep 3
  # hard-kill stragglers still holding the ports
  local pid port
  for port in 8200 8201 8202 8203; do
    for pid in $(ss -ltnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | sort -u); do
      kill -KILL "$pid" 2>/dev/null
    done
  done
}

start_agent_vllm() {  # $1 = model_name , $2 = model_path
  local name="$1" path="$2" i port v
  log "starting vLLM x4 for $name (served-model-name=$name)"
  for i in 0 1 2 3; do
    port=$((8200 + i))
    if vllm_up "$port"; then log "vLLM :$port already up"; continue; fi
    CUDA_VISIBLE_DEVICES=$i setsid nohup "$VLLM" serve "$path" \
      --host 127.0.0.1 --port "$port" --served-model-name "$name" \
      --gpu-memory-utilization 0.70 --max-model-len 32768 --max-num-seqs 256 \
      --tensor-parallel-size 1 --dtype bfloat16 --no-enable-log-requests --max-logprobs 40 \
      > "/tmp/vllm_${port}.log" 2>&1 &
  done
  for i in $(seq 1 60); do
    v=0; for port in 8200 8201 8202 8203; do vllm_up "$port" && v=$((v+1)); done
    [ "$v" = 4 ] && { log "vLLM x4 UP for $name"; return 0; }
    sleep 5
  done
  log "FATAL: vLLM for $name not all up (v=$v/4) — see /tmp/vllm_*.log"
  return 1
}

# ── arm / probe primitives (model-agnostic; EVOLVER_AGENT_MODEL picks agent) ──
run_alf_arm() {  # $1 = tag ; rest = extra args
  local tag="$1"; shift
  log "=== ALF ARM $tag ==="
  "$PY" -m recipe.agent_evolver.run alfworld --split heldin --num-rounds 6 --num-tasks 64 \
    --run-tag "$tag" --agent-api-bases "$APIS" --concurrency "$CONC" --env-urls "$ALF_HELDIN_ENV" "$@"
  log "=== ALF ARM DONE $tag (exit $?) ==="
}

run_ws_arm() {  # $1 = tag ; rest = extra args
  local tag="$1"; shift
  log "=== WS ARM $tag ==="
  "$PY" -m recipe.agent_evolver.run webshop --num-rounds 6 --num-tasks 64 \
    --start 1500 --seed 0 --env-urls "$WS_ENV_URL" --concurrency "$CONC" \
    --run-tag "$tag" --agent-api-bases "$APIS" "$@"
  log "=== WS ARM DONE $tag (exit $?) ==="
}

alf_probe() {  # $1 = tag , $2 = base-config yaml
  local tag="$1" cfg="$2"
  log "=== ALF HELDOUT $tag <- $cfg ==="
  "$PY" -m recipe.agent_evolver.run alfworld --split heldout --num-rounds 1 --num-tasks 64 \
    --run-tag "$tag" --agent-api-bases "$APIS" --concurrency "$CONC" --env-urls "$ALF_HELDOUT_ENV" --base-config "$cfg"
  log "=== ALF HELDOUT DONE $tag (exit $?) ==="
}

ws_probe() {  # $1 = tag , $2 = base-config yaml
  local tag="$1" cfg="$2"
  log "=== WS HELDOUT $tag <- $cfg ==="
  "$PY" -m recipe.agent_evolver.run webshop --num-rounds 1 --num-tasks 64 \
    --start 0 --seed 0 --env-urls "$WS_ENV_URL" --concurrency "$CONC" \
    --run-tag "$tag" --agent-api-bases "$APIS" --base-config "$cfg"
  log "=== WS HELDOUT DONE $tag (exit $?) ==="
}

best_config() {  # $1 = comparison.json -> prints R<n>/config.yaml (or nothing)
  [ -f "$1" ] || return 0
  "$PY" - "$1" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
best = max(d["rounds"], key=lambda r: r["mean_reward"])
print(f"R{best['round']}/config.yaml")
PYEOF
}

# ── one benchmark cell: llm/prior/enforce held-in + 4 held-out probes ─────────
run_alf_cell() {  # $1 = tag-prefix (e.g. ov3x_2b)
  local P="$1"
  run_alf_arm "${P}_llm_heldin"
  if kev_up; then run_alf_arm "${P}_prior_heldin" --decision-backend kev --decision-mode prior; else log "kev DOWN — skip prior"; fi
  if kev_up; then run_alf_arm "${P}_enforce_heldin" --decision-backend kev --decision-mode enforce; else log "kev DOWN — skip enforce"; fi

  alf_probe "${P}_baseline_heldout" "$RUNS/${P}_llm_heldin/R0/config.yaml"
  local LB; LB=$(best_config "$RUNS/${P}_llm_heldin/comparison.json"); [ -n "$LB" ] && alf_probe "${P}_llmbest_heldout" "$RUNS/${P}_llm_heldin/$LB"
  local PB; PB=$(best_config "$RUNS/${P}_prior_heldin/comparison.json"); [ -n "$PB" ] && alf_probe "${P}_priorbest_heldout" "$RUNS/${P}_prior_heldin/$PB"
  local EB; EB=$(best_config "$RUNS/${P}_enforce_heldin/comparison.json"); [ -n "$EB" ] && alf_probe "${P}_enforcebest_heldout" "$RUNS/${P}_enforce_heldin/$EB"
}

run_ws_cell() {  # $1 = tag-prefix (e.g. ws3x_2b)
  local P="$1"
  run_ws_arm "${P}_llm_heldin"
  if kev_up; then run_ws_arm "${P}_prior_heldin" --decision-backend kev --decision-mode prior; else log "kev DOWN — skip prior"; fi
  if kev_up; then run_ws_arm "${P}_enforce_heldin" --decision-backend kev --decision-mode enforce; else log "kev DOWN — skip enforce"; fi

  ws_probe "${P}_baseline_heldout" "$RUNS/${P}_llm_heldin/R0/config.yaml"
  local LB; LB=$(best_config "$RUNS/${P}_llm_heldin/comparison.json"); [ -n "$LB" ] && ws_probe "${P}_llmbest_heldout" "$RUNS/${P}_llm_heldin/$LB"
  local PB; PB=$(best_config "$RUNS/${P}_prior_heldin/comparison.json"); [ -n "$PB" ] && ws_probe "${P}_priorbest_heldout" "$RUNS/${P}_prior_heldin/$PB"
  local EB; EB=$(best_config "$RUNS/${P}_enforce_heldin/comparison.json"); [ -n "$EB" ] && ws_probe "${P}_enforcebest_heldout" "$RUNS/${P}_enforce_heldin/$EB"
}

cell_done() {  # $1 = tag-prefix, $2 = benchmark (alf|ws) -> 0 if final probe comparison.json exists
  local P="$1"
  [ -f "$RUNS/${P}_enforcebest_heldout/comparison.json" ] && return 0 || return 1
}

# ── infra guards ─────────────────────────────────────────────────────────────
ensure_env() {
  local p
  for p in $(seq 18100 18163); do
    if env_up "$p"; then continue; fi
    ALFWORLD_DATA="$ALF_DATA" ALFWORLD_SPLIT=eval_in_distribution \
      setsid nohup "$ALF_PY" "$REPO/benchmarks/alfworld/env_server.py" --port "$p" \
      > "/tmp/alfworld_heldin_$p.log" 2>&1 &
    log "ALF held-in :$p launching"
  done
  for p in $(seq 18200 18263); do
    if env_up "$p"; then continue; fi
    ALFWORLD_DATA="$ALF_DATA" ALFWORLD_SPLIT=eval_out_of_distribution \
      setsid nohup "$ALF_PY" "$REPO/benchmarks/alfworld/env_server.py" --port "$p" \
      > "/tmp/alfworld_heldout_$p.log" 2>&1 &
    log "ALF held-out :$p launching"
  done
  if env_up 18090; then log "WS :18090 already up"; else
    log "WS :18090 launching (official full set, seed 233 — ~1-2 min)"
    JAVA_HOME="$WS_JAVA_HOME" PATH="$WS_JAVA_HOME/bin:$PATH" \
      WEBSHOP_ROOT="$WS_ROOT" \
      WEBSHOP_NUM_PRODUCTS=full WEBSHOP_HUMAN_GOALS=1 WEBSHOP_SEED=233 \
      WEBSHOP_FILE_PATH="$WS_DATA/items_shuffle.json" \
      WEBSHOP_ATTR_PATH="$WS_DATA/items_ins_v2.json" \
      setsid nohup "$WS_PY" "$REPO/benchmarks/webshop/env_server.py" --host 127.0.0.1 --port 18090 \
      > /tmp/webshop_env_18090.log 2>&1 &
  fi

  # wait for ALF env servers (64 held-in + 64 held-out) to be healthy
  local _i _n
  for _i in $(seq 1 60); do
    _n=0
    for p in $(seq 18100 18163) $(seq 18200 18263); do env_up "$p" && _n=$((_n+1)); done
    [ "$_n" = 128 ] && break
    sleep 5
  done
  log "ALF env servers ready: $_n/128"
}

# ── main ─────────────────────────────────────────────────────────────────────
wait_for_models() {  # block until /tmp/model_download.log reports ALL DONE
  log "waiting for model downloads to finish (poll /tmp/model_download.log)"
  local i
  for i in $(seq 1 360); do   # up to ~2h
    if grep -q "ALL DONE" /tmp/model_download.log 2>/dev/null; then
      log "model downloads complete"; return 0
    fi
    sleep 20
  done
  log "FATAL: model download did not finish in time"; return 1
}

cd "$REPO" || { echo "no repo" >&2; exit 1; }
log "three_axis scale sweep start (2B then 0.8B, ALF then WS)"

if ! kev_up; then
  log "FATAL: kev :8090 DOWN — prior/enforce arms need it. Abort."
  exit 1
fi
wait_for_models || { log "ABORT: models not ready"; exit 1; }
ensure_env

run_model() {  # $1 = model_name , $2 = model_path , $3 = alf-tag , $4 = ws-tag
  local name="$1" path="$2" atag="$3" wtag="$4"
  [ -d "$path" ] || { log "FATAL: model path missing: $path"; return 1; }
  log "########## MODEL $name ##########"
  export EVOLVER_AGENT_MODEL="$name"

  stop_agent_vllm
  start_agent_vllm "$name" "$path" || { log "ABORT model $name (vLLM failed)"; return 1; }

  if cell_done "$atag"; then log "SKIP ALF $name (already done)"; else run_alf_cell "$atag"; fi
  if cell_done "$wtag";  then log "SKIP WS  $name (already done)"; else run_ws_cell  "$wtag"; fi
}

run_model Qwen3.5-2B   /mnt/llmshared-ssd-hd/shishuqing/models/Qwen3.5-2B   ov3x_2b   ws3x_2b
run_model Qwen3.5-0.8B /mnt/llmshared-ssd-hd/shishuqing/models/Qwen3.5-0.8B ov3x_0_8b ws3x_0_8b

# ── final summary ────────────────────────────────────────────────────────────
log "=== FINAL SUMMARY ==="
for P in ov3x_2b ws3x_2b ov3x_0_8b ws3x_0_8b; do
  for t in "${P}_llm_heldin" "${P}_prior_heldin" "${P}_enforce_heldin" \
           "${P}_baseline_heldout" "${P}_llmbest_heldout" "${P}_priorbest_heldout" "${P}_enforcebest_heldout"; do
    cmp="$RUNS/$t/comparison.json"
    if [ -f "$cmp" ]; then
      log "--- $t ---"
      "$PY" - "$cmp" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
for r in d["rounds"]:
    print(f"  R{r['round']} {r['config']}: pass={r['passed']}/{r['tasks']} rate={r['pass_rate']} reward={r['mean_reward']}")
PYEOF
    else
      log "$t: NO comparison.json (failed or skipped)"
    fi
  done
done
log "=== SCALE SWEEP COMPLETE ==="
