#!/usr/bin/env bash
# =============================================================================
# Harness-JEV — multi-pod multi-GPU evolver launcher (all benchmarks, cloudml)
# =============================================================================
# Runs on a Volcano/K8s training job where the platform injects RANK/WORLD_SIZE
# and runs THIS SAME command on every pod (master-0, worker-0, worker-1, …),
# each pod with 4 local GPUs.  Every pod serves the same agent model (Qwen3.5-4B)
# on its local GPUs; RANK 0 additionally runs the evolver and shards the rollout
# across ALL pods' vLLM endpoints (data-parallel).  Model weights are never
# modified — this only fans the rollout out.
#
#   pod layout (Volcano):  RANK 0 = master-0 (orchestrator + 4 GPUs)
#                          RANK i = worker-{i-1} (4 GPUs each, serve + wait)
#
# Benchmarks (set BENCH):
#   frozenlake | sokoban   → run_gridgames.py   (env is in-process, no env server)
#   alfworld               → run.py alfworld     (stateful ALF env servers, one per GPU on every pod)
#   webshop                → run.py webshop      (single WebShop env server, on RANK 0)
#
# Every parameter is a DEFAULT; override any of them via an env var of the same
# name, e.g.:  BENCH=alfworld SPLIT=heldout ROUNDS=3 bash run_multinode.sh
#
# Pod discovery (orchestrator RANK 0 only):
#   - NODE_ADDRS="ip0,ip1,ip2"  (optional) explicit comma-separated pod addresses
#     in rank order — set this if worker-{i} short hostnames do NOT resolve.
#   - otherwise rank 0 = $MASTER_ADDR, rank i = worker-{i-1} (K8s pod DNS).
#
# Requirements
#   - identical REPO_DIR / VLLM_BIN / MODEL_PATH on every pod (shared JuiceFS).
#   - vLLM ports (PORT_BASE .. PORT_BASE+GPUS-1) reachable across pods (no netpol
#     block; bind 0.0.0.0).
#   - ANTHROPIC_API_KEY + ANTHROPIC_BASE_URL exported on RANK 0 only (the meta
#     model deepseek-v4-pro runs there; only the orchestrator needs egress).
#   - for alfworld/webshop: env-server conda envs + data paths exist on EVERY pod
#     (ALF_PY/WS_PY/ALF_DATA/WS_DATA — ALF serves one env per GPU on each pod);
#     START_ENV_SERVERS=1 (default) brings them up, START_ENV_SERVERS=0 assumes up.
#   - cluster nodes lack python3.10-dev + CUDA toolchain: the script preflights
#     Python.h/gcc and routes around FlashInfer JIT (see below).
#
# Usage (same command on every pod; platform injects RANK/WORLD_SIZE)
#   bash run_multinode.sh run        # env servers (R0) → vLLM → evolve → teardown
#   bash run_multinode.sh start      # serve vLLM only, print this pod's endpoints
#   bash run_multinode.sh status     # probe every endpoint's /v1/models (RANK 0)
#   bash run_multinode.sh teardown   # kill this pod's vLLM servers
#
#   Run `run` inside tmux/screen — the evolution is long-lived (~hours).
# =============================================================================
set -euo pipefail

# ── cluster (platform-injected) ───────────────────────────────────────────────
RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NODE_ADDRS=${NODE_ADDRS:-}              # optional explicit pod addresses, rank order
PORT_BASE=8200                          # pod-local port = PORT_BASE + gpu_idx
MAX_GPUS_PER_NODE=0                     # 0 = auto-detect (nvidia-smi -L | wc -l), else cap

# ── shared paths (every pod) ──────────────────────────────────────────────────
REPO_DIR="/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV"
VLLM_BIN="/mnt/llmshared-ssd-hd/shishuqing/.venv/bin/vllm"
MODEL_PATH="/mnt/llmshared-ssd-hd/shishuqing/models/Qwen3.5-4B"
MODEL_NAME="Qwen3.5-4B"
PYINC_BUNDLE="/mnt/llmshared-ssd-hd/shishuqing/python310-include"   # cluster has no python3.10-dev

# ── env-server paths (ALF: every pod; WS: RANK 0 only) ────────────────────────
ALF_PY="/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-alfworld/bin/python"
WS_PY="/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-webshop/bin/python"
WS_JAVA_HOME="/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-webshop/lib/jvm"
ALF_DATA="/mnt/llmshared-ssd-hd/cty/data/alfworld"
WS_DATA="/mnt/llmshared-ssd-hd/chenjinyuan/memory-agent/data/webshop_official_64fa2a5"

# ── vLLM flags ────────────────────────────────────────────────────────────────
GPU_MEM_UTIL=0.85
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=128                        # Qwen3.5 Mamba-hybrid: large seq counts overflow Mamba cache
VLLM_LOG_DIR="/tmp"
READY_TIMEOUT_S=1200
START_ENV_SERVERS=1                     # 1 = RANK 0 brings up env servers (ALF/WS); 0 = assume up
# ──────────────────────────────────────────────────────────────────────────────

# ── benchmark selection + per-benchmark defaults (all env-overridable) ────────
# BENCH: frozenlake | sokoban | alfworld | webshop
BENCH="${BENCH:-sokoban}"

ROUNDS="${ROUNDS:-6}"
case "$BENCH" in
  frozenlake|sokoban)
    TIER="${TIER:-L16}"                       # L4|L8|L16|L32
    SPLIT="${SPLIT:-evolve}"                  # evolve|gate|test
    EPISODES_PER_LEVEL="${EPISODES_PER_LEVEL:-4}"
    NUM_TASKS="${NUM_TASKS:-0}"               # 0 = all levels in the split
    CONCURRENCY="${CONCURRENCY:-48}"          # per-endpoint concurrency
    ENV_URLS=""                               # no env server (in-process ta_env)
    ;;
  alfworld)
    SPLIT="${SPLIT:-heldin}"                  # heldin|heldout
    NUM_TASKS="${NUM_TASKS:-64}"
    START="${START:-0}"                       # game_idx offset
    SEED="${SEED:-1234}"
    CONCURRENCY="${CONCURRENCY:-}"            # computed = WORLD_SIZE×GPUs (one worker per env server)
    ENV_URLS=""                               # computed cross-pod at runtime (one env server per GPU)
    ALF_ENV_PORT_BASE="${ALF_ENV_PORT_BASE:-}"
    if [ "$SPLIT" = heldout ]; then
      ALF_ENV_PORT_BASE="${ALF_ENV_PORT_BASE:-18086}"
    else
      ALF_ENV_PORT_BASE="${ALF_ENV_PORT_BASE:-18082}"
    fi
    ;;
  webshop)
    NUM_TASKS="${NUM_TASKS:-64}"
    START="${START:-1500}"                    # train slice 1500+; use 0 for the test probe
    SEED="${SEED:-0}"
    CONCURRENCY="${CONCURRENCY:-}"            # empty → run.py defaults to len(api_bases)
    ENV_URLS="${ENV_URLS:-http://127.0.0.1:18090}"
    ;;
  *)
    echo "ERROR: unknown BENCH='$BENCH' (use frozenlake|sokoban|alfworld|webshop)" >&2
    exit 2
    ;;
esac
RUN_TAG="${RUN_TAG:-${BENCH}_$(date +%Y%m%d_%H%M%S)}"
BASE_CONFIG="${BASE_CONFIG:-}"                # optional: start from an evolved config (held-out probe)
DONE_FILE="$REPO_DIR/recipe/agent_evolver/multi_node/.done.$RUN_TAG"
# ──────────────────────────────────────────────────────────────────────────────

# cluster nodes lack python3.10-dev → point gcc at bundled headers (serve_vllm.sh)
if [ -d "$PYINC_BUNDLE" ] && [ ! -d /usr/include/python3.10 ]; then
  export CPATH="$PYINC_BUNDLE${CPATH:+:$CPATH}"
fi
# no CUDA toolchain (no ninja/nvcc) → route around FlashInfer's runtime JIT
export VLLM_USE_FLASHINFER_SAMPLER=0

VLLM_FLAGS="--host 0.0.0.0 --served-model-name $MODEL_NAME \
--gpu-memory-utilization $GPU_MEM_UTIL --max-model-len $MAX_MODEL_LEN \
--max-num-seqs $MAX_NUM_SEQS --tensor-parallel-size 1 --dtype bfloat16 \
--gdn-prefill-backend triton --no-enable-log-requests --max-logprobs 40"

log()  { echo "[multinode][R$RANK] $(date '+%H:%M:%S') $*"; }
die()  { echo "[multinode][R$RANK] ERROR: $*" >&2; exit 1; }
vllm_up() { curl -s -m 3 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
env_up()  { curl -s -m 3 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }

# ── per-pod preflight (fail fast on known cluster-image gaps) ─────────────────
preflight() {
  local _fail=0
  if [ ! -f /usr/include/python3.10/Python.h ] && [ ! -f "$PYINC_BUNDLE/Python.h" ]; then
    echo "!! missing Python.h (vLLM model-inspection compile needs it): install python3.10-dev or bundle headers"; _fail=1
  fi
  command -v gcc >/dev/null 2>&1 || { echo "!! missing gcc (vLLM model-inspection compile needs it)"; _fail=1; }
  [ -x "$VLLM_BIN" ] || { echo "!! missing vLLM bin: $VLLM_BIN"; _fail=1; }
  [ -d "$MODEL_PATH" ] || { echo "!! missing model dir: $MODEL_PATH"; _fail=1; }
  [ "$_fail" -ne 0 ] && die "preflight failed — fix above and re-run"
  log "preflight OK (Python.h + gcc + vLLM + model present)"
}

# ── pod-local GPU count (no ssh) ──────────────────────────────────────────────
gpu_count() {
  local n
  n=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [ "${MAX_GPUS_PER_NODE:-0}" -gt 0 ]; then
    [ "$n" -gt "$MAX_GPUS_PER_NODE" ] && n="$MAX_GPUS_PER_NODE"
  fi
  echo "$n"
}

# ── pod address for a global rank (0..WORLD_SIZE-1) ───────────────────────────
node_host() {
  local r="$1"
  if [ -n "$NODE_ADDRS" ]; then
    echo "$NODE_ADDRS" | cut -d, -f"$((r + 1))"
  elif [ "$r" = "0" ]; then
    echo "$MASTER_ADDR"
  else
    echo "worker-$((r - 1))"
  fi
}

# ── vLLM serve (this pod's local GPUs only) ───────────────────────────────────
start_local() {
  local ngpu gpu port logf
  ngpu=$(gpu_count)
  [ "$ngpu" -gt 0 ] || die "this pod reports 0 GPUs"
  log "pod host=$(hostname) rank=$RANK: starting $ngpu vLLM server(s) on ports $PORT_BASE..$((PORT_BASE + ngpu - 1))"
  for (( gpu = 0; gpu < ngpu; gpu++ )); do
    port=$((PORT_BASE + gpu))
    logf="$VLLM_LOG_DIR/vllm_${MODEL_NAME}_r${RANK}_${port}.log"
    CUDA_VISIBLE_DEVICES=$gpu setsid "$VLLM_BIN" serve "$MODEL_PATH" $VLLM_FLAGS --port "$port" \
      >"$logf" 2>&1 </dev/null &
  done
}

# every pod endpoint (rank r → gpu g): used by RANK 0 to build --agent-api-bases
all_endpoints() {
  local r g ngpu out=""
  ngpu=$(gpu_count)   # uniform per pod (same pod spec)
  for (( r = 0; r < WORLD_SIZE; r++ )); do
    for (( g = 0; g < ngpu; g++ )); do
      out="${out}${out:+,}http://$(node_host $r):$((PORT_BASE + g))/v1"
    done
  done
  echo "$out"
}

# cross-pod ALFWorld env-server URLs (one per GPU, port ALF_ENV_PORT_BASE+g)
# — same ordering as all_endpoints, so worker i pairs vLLM endpoint i with env i.
alf_env_urls() {
  local r g ngpu out=""
  ngpu=$(gpu_count)
  for (( r = 0; r < WORLD_SIZE; r++ )); do
    for (( g = 0; g < ngpu; g++ )); do
      out="${out}${out:+,}http://$(node_host $r):$((ALF_ENV_PORT_BASE + g))"
    done
  done
  echo "$out"
}

wait_ready() {
  log "waiting for all $((WORLD_SIZE * $(gpu_count))) endpoints (timeout ${READY_TIMEOUT_S}s)"
  local eps url waited code
  eps=$(all_endpoints)
  IFS=',' read -ra urls <<< "$eps"
  for url in "${urls[@]}"; do
    waited=0
    while true; do
      code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$url/v1/models" 2>/dev/null || echo 000)
      [ "$code" = "200" ] && break
      waited=$((waited + 5))
      [ "$waited" -ge "$READY_TIMEOUT_S" ] && die "endpoint $url never became ready (last=$code)"
      sleep 5
    done
    log "ready: $url"
  done
}

status() {
  local url code
  IFS=',' read -ra urls <<< "$(all_endpoints)"
  for url in "${urls[@]}"; do
    code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$url/v1/models" 2>/dev/null || echo 000)
    echo "  $url/v1 -> $code"
  done
}

teardown() {
  log "killing this pod's vLLM servers matching MODEL_PATH=$MODEL_PATH"
  pkill -f "vllm serve $MODEL_PATH" 2>/dev/null || true
  sleep 3
  log "teardown done"
}

# ── env servers (ALF: every pod; WS: RANK 0 only) ────────────────────────────

start_env_servers() {
  case "$BENCH" in frozenlake|sokoban) return 0 ;; esac
  [ "$START_ENV_SERVERS" = "1" ] || { log "START_ENV_SERVERS=0 — assuming env servers already up"; return 0; }

  local p split_env waited ready g
  if [ "$BENCH" = alfworld ]; then
    # ALF env is single-instance stateful → ONE server per GPU on EVERY pod
    # (bound 0.0.0.0 so RANK 0 can reach them cross-pod): 16 GPUs → 16 servers.
    [ "$SPLIT" = heldout ] && split_env=eval_out_of_distribution || split_env=eval_in_distribution
    for (( g = 0; g < $(gpu_count); g++ )); do
      p=$((ALF_ENV_PORT_BASE + g))
      if env_up "$p"; then log "ALF env :$p already up (this pod)"; continue; fi
      ALFWORLD_DATA="$ALF_DATA" ALFWORLD_SPLIT="$split_env" \
        setsid nohup "$ALF_PY" "$REPO_DIR/benchmarks/alfworld/env_server.py" --host 0.0.0.0 --port "$p" \
        > "/tmp/alfworld_${split_env}_r${RANK}_${p}.log" 2>&1 &
      log "launched ALF env :$p ($split_env) on this pod"
    done
  else
    # WebShop: single server on RANK 0 (fresh uuid per task absorbs many workers)
    [ "$RANK" = "0" ] || return 0
    p="${WS_ENV_PORT:-18090}"
    if env_up "$p"; then log "WS env :$p already up"; return 0; fi
    JAVA_HOME="$WS_JAVA_HOME" PATH="$WS_JAVA_HOME/bin:$PATH" \
      WEBSHOP_ROOT="$REPO_DIR/benchmarks/webshop/official_webshop" \
      WEBSHOP_NUM_PRODUCTS=full WEBSHOP_HUMAN_GOALS=1 WEBSHOP_SEED=233 \
      WEBSHOP_FILE_PATH="$WS_DATA/items_shuffle.json" \
      WEBSHOP_ATTR_PATH="$WS_DATA/items_ins_v2.json" \
      setsid nohup "$WS_PY" "$REPO_DIR/benchmarks/webshop/env_server.py" --port "$p" \
      > "/tmp/webshop_env_server_${p}.log" 2>&1 &
    log "launched WS env :$p"
  fi

  # wait for THIS pod's local env servers to become healthy
  waited=0
  while true; do
    ready=1
    if [ "$BENCH" = alfworld ]; then
      for (( g = 0; g < $(gpu_count); g++ )); do env_up $((ALF_ENV_PORT_BASE + g)) || ready=0; done
    else
      env_up "${WS_ENV_PORT:-18090}" || ready=0
    fi
    [ "$ready" = "1" ] && break
    waited=$((waited + 5))
    [ "$waited" -ge 300 ] && die "env servers never became healthy (check /tmp/*_env*.log)"
    sleep 5
  done
  log "env servers healthy on this pod"
}

# ── dispatch ──────────────────────────────────────────────────────────────────

run_evolution() {
  local eps ntotal env_urls conc
  eps=$(all_endpoints)
  ntotal=$(echo "$eps" | tr ',' '\n' | wc -l)
  log "orchestrator: BENCH=$BENCH × $ROUNDS rounds across $ntotal model endpoints"
  log "model endpoints: $eps"
  cd "$REPO_DIR"

  # ALFWorld: env URLs + concurrency scale with the cluster (one env server per GPU)
  if [ "$BENCH" = alfworld ]; then
    env_urls="$(alf_env_urls)"
    conc="${CONCURRENCY:-$((WORLD_SIZE * $(gpu_count)))}"
  else
    env_urls="$ENV_URLS"
    conc="$CONCURRENCY"
  fi
  [ -n "$env_urls" ] && log "env servers: $env_urls (concurrency=$conc)"

  local -a CMD
  case "$BENCH" in
    frozenlake|sokoban)
      CMD=( .venv/bin/python -m recipe.agent_evolver.run_gridgames
            --game "$BENCH" --tier "$TIER" --split "$SPLIT" --num-rounds "$ROUNDS"
            --episodes-per-level "$EPISODES_PER_LEVEL"
            --agent-api-bases "$eps" --concurrency "$CONCURRENCY" --run-tag "$RUN_TAG" )
      [ "$NUM_TASKS" != "0" ] && CMD+=( --num-tasks "$NUM_TASKS" )
      ;;
    alfworld|webshop)
      CMD=( .venv/bin/python -m recipe.agent_evolver.run "$BENCH"
            --num-rounds "$ROUNDS" --num-tasks "$NUM_TASKS" --start "$START" --seed "$SEED"
            --agent-api-bases "$eps" --env-urls "$env_urls" --run-tag "$RUN_TAG" )
      [ "$BENCH" = alfworld ] && CMD+=( --split "$SPLIT" )
      [ -n "$conc" ] && CMD+=( --concurrency "$conc" )
      ;;
  esac
  [ -n "$BASE_CONFIG" ] && CMD+=( --base-config "$BASE_CONFIG" )

  log "exec: ${CMD[*]}"
  "${CMD[@]}" 2>&1 | tee "/tmp/${RUN_TAG}.log"
}

# ── subcommands ───────────────────────────────────────────────────────────────

case "${1:-run}" in
  run)
    preflight
    start_local
    start_env_servers            # ALF: every pod; WS: RANK 0 only; gridgames: no-op
    if [ "$RANK" = "0" ]; then
      wait_ready
      run_evolution
      teardown
      touch "$DONE_FILE"
      log "orchestrator done → $DONE_FILE"
    else
      log "worker: serving vLLM; waiting for orchestrator done-file $DONE_FILE"
      while [ ! -f "$DONE_FILE" ]; do sleep 30; done
      teardown
      log "worker: orchestrator finished — exiting"
    fi
    ;;
  start)    preflight; start_local; wait_ready 2>/dev/null || true; echo "ENDPOINTS=$(all_endpoints)" ;;
  status)   status ;;
  teardown) teardown ;;
  *)        die "unknown subcommand '$1' (use run|start|status|teardown)" ;;
esac
