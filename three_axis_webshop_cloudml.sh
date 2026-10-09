#!/usr/bin/env bash
# =============================================================================
# three_axis_webshop_cloudml.sh — kev 三轴对照 (llm vs prior vs enforce) 在 WebShop 上，cloudml 16 卡任务
# =============================================================================
# 与 three_axis_cloudml.sh（ALFWorld 版）同构，只是 benchmark/env-server/held-in-out 语义换成 WebShop：
#
#   RANK 0 (master-0) -> llm arm（无 kev 决策）→ 接着 enforce arm（kev）→ join → 4 探针 + summary
#   RANK 1 (worker-0) -> prior arm（kev）
#
# 两个 pod 各自独立在【本地】卡上干活（vLLM 绑 127.0.0.1，不跨 pod 网络），结果落到共享 JuiceFS。
# 两个节点都要起 kev :8090（R0 给 enforce、R1 给 prior），都挤在 GPU 0（vLLM gpu-mem 0.70）。
#
# WebShop 与 ALFWorld 的 held-in/held-out 语义【不同】：
#   - WebShop 是【同一个官方 human goal 池】(12087 目标, seed 233, env :18090)，没有独立 split。
#   - held-in（进化） = 官方 train 切片  goals 1500–1563  →  run.py --start 1500
#   - held-out（只测）= 官方 test 切片   goals 0–63       →  run.py --start 0
#   （方法学 2026-10-01 拍板：放弃 synthetic/human，改官方 train→test 真 OOD 协议，见
#     memory webshop-heldin-heldout-split / harnessx-jev-webshop-evolve 第 31 条。）
#
#   所以 WebShop 只需【每 pod 一个 env server】(:18090，ThreadingHTTPServer 线程安全、每次
#   /create 独立 uuid 实例可吸收 N 个并发 worker)，held-in/held-out 靠 run.py 的 --start 切，
#   RANK 0 探针阶段【无需再起第二个 env server】。这是与 ALFWorld 脚本最大的结构差异。
#
# -----------------------------------------------------------------------------
# 提交（在 CloudML 平台填这些，其余都在本脚本内默认好）：
#   pod 数：     2（1 master + 1 worker）
#   每 pod GPU： RESOURCE_GPU=8（M402）→ 共 16 卡；脚本自动读，缺省 nvidia-smi
#   Docker 命令： bash /mnt/llmshared-ssd-hd/shishuqing/Harness-JEV/three_axis_webshop_cloudml.sh
#   meta-agent 模型：多机时自动用 pod 挂载的 /preset-models/public/DeepSeek-V4.1-Flash
#       （本地 vLLM serve :8400），无需 ANTHROPIC_* env；若该路径缺失（本地冒烟）才回退
#       anthropic 网关（此时才需要 ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL）。
#   可选 env（覆盖默认值）： ROUNDS=6 NUM_TASKS=64 SEED=0 RUN_TAG=... GPU_MEM_UTIL=0.70
#       META_TP=1 META_GPU_MEM_UTIL=0.90 META_MAX_MODEL_LEN=32768
# -----------------------------------------------------------------------------
#
# 每 pod 本地布局（N = RESOURCE_GPU，preset meta 时 agent vLLM = AGENT_N = N - META_TP）：
#   vLLM ×AGENT_N        :8200..8200+AGENT_N-1   (Qwen3.5-4B, gpu-mem 0.70)
#   meta vLLM ×1         :8400               (DeepSeek-V4.1-Flash, 最后 META_TP 张卡)
#   kev ×1               :8090               (4B-Base+LoRA+head, GPU 0)
#   WebShop ×1           :18090              (官方 full 集, 12087 human goals, seed 233)
#
# 日志：结果在 runs/evolve/<RUN_TAG>_<arm>_heldin/（held-out 探针 <RUN_TAG>*_heldout）；
# 调试日志（vLLM/kev/env）在 runs/logs/<RUN_TAG>/（共享盘，pod 回收不丢，r$RANK 前缀区分）。
#
# teardown 是精确 PID（nohup 记 $! 进 pidfile，只杀本 pod 自己起的进程），不用 pkill -f。
# =============================================================================
set -uo pipefail

# ── 平台注入（cloudml）────────────────────────────────────────────────────────
RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-2}
MASTER_ADDR=${MASTER_ADDR:-}            # 本 job 不跨 pod 通信，仅占位
MASTER_PORT=${MASTER_PORT:-29517}       # 同上
RESOURCE_GPU=${RESOURCE_GPU:-}          # 空 = 自动 nvidia-smi

# ── 共享路径（每台 pod 相同）─────────────────────────────────────────────────
REPO_DIR="/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV"
PY="$REPO_DIR/.venv/bin/python"
VLLM_BIN="/mnt/llmshared-ssd-hd/shishuqing/.venv/bin/vllm"
MODEL_PATH="/mnt/llmshared-ssd-hd/shishuqing/models/Qwen3.5-4B"
MODEL_NAME="Qwen3.5-4B"
KEV_REPO="/mnt/llmshared-ssd-hd/shishuqing/kev"
KEV_RUN="/mnt/llmshared-ssd-hd/shishuqing/models/kev-4b"
# WebShop env server（官方 full 集；ee-webshop conda env 有 Java + spacy）
WS_PY="/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-webshop/bin/python"
WS_JAVA_HOME="/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-webshop/lib/jvm"
WS_DATA="/mnt/llmshared-ssd-hd/chenjinyuan/memory-agent/data/webshop_official_64fa2a5"
WS_ROOT="$REPO_DIR/benchmarks/webshop/official_webshop"
PYINC_BUNDLE="/mnt/llmshared-ssd-hd/shishuqing/python310-include"
# kev 的 base 模型（Qwen3.5-4B-Base）：kev.serve 按 hub id 从 HF cache 解析，集群 pod 的
# HF cache 是空的 → 启动前 symlink 到共享盘本地模型目录（见 kev_hf_cache）。
KEV_BASE_HUB_ID="Qwen/Qwen3.5-4B-Base"
KEV_BASE_REV="1001bb4d826a52d1f399e183466143f4da7b741b"
KEV_BASE_DIR="/mnt/llmshared-ssd-hd/shishuqing/models/Qwen3.5-4B-Base"

# ── 端口（pod 内）─────────────────────────────────────────────────────────────
PORT_BASE=8200                          # vLLM
KEV_PORT=8090                           # kev（两节点都要）
WS_ENV_PORT=18090                       # WebShop env server（每 pod 一个）

# ── meta-agent 模型：多机 pod 够不到 mioffice anthropic 网关（api.llm.mioffice.cn
#    只解析到内网 10.x），改用 pod 挂载的 preset 模型本地 vLLM serve。本机/本地冒烟
#    时该路径不存在 → 自动回退 anthropic 网关（本地可达）。─────────────────────
META_MODEL_PATH="/preset-models/public/DeepSeek-V4.1-Flash"
META_MODEL_NAME="DeepSeek-V4.1-Flash"
META_PORT=8400                          # meta vLLM 专用端口（agent vLLM 在 8200..）
META_TP=${META_TP:-1}                   # meta 模型 tensor-parallel 卡数（大 MoE 上调）
META_GPU_MEM_UTIL=${META_GPU_MEM_UTIL:-0.90}
META_MAX_MODEL_LEN=${META_MAX_MODEL_LEN:-32768}

# ── vLLM 参数 ────────────────────────────────────────────────────────────────
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.70}      # 0.70（非 0.85）给 GPU0 上 kev 让出 ~19GiB
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}       # rollout 每 req 最多 ~1024 tok，256 够用

# ── 实验参数（env 可覆盖）─────────────────────────────────────────────────────
ROUNDS=${ROUNDS:-6}
NUM_TASKS=${NUM_TASKS:-64}
SEED=${SEED:-0}                         # webshop 默认（human goal 顺序由 env seed=233 决定，run 层 seed 不影响 goal）
START_TRAIN=${START_TRAIN:-1500}        # held-in 进化切片 = 官方 train goals 1500..1563
START_TEST=${START_TEST:-0}             # held-out 探针切片 = 官方 test goals 0..63
RUNS="$REPO_DIR/recipe/agent_evolver/runs/evolve"
DONE_DIR="$REPO_DIR/recipe/agent_evolver/multi_node"   # 共享盘 join 屏障
PIDFILE="/tmp/three_axis_ws_pids_r$RANK"   # 本 pod 起的精确 PID

export EVOLVER_AGENT_TEMPERATURE=0      # 确定性 rollout（贪心）

# ── 共享 RUN_TAG：两 pod 必须一致，否则 join 屏障/结果目录对不上。优先 MASTER_ADDR
# （平台注入、两 pod 相同，形如 tj-<job>-master-0），剥 -master-N/-worker-N 后缀；
# 再试本 pod hostname；都不匹配才退回时间戳（不一致，仅单机调试用）。
if [ -z "${RUN_TAG:-}" ]; then
  RUN_TAG=""
  for _h in "${MASTER_ADDR:-}" "$(hostname 2>/dev/null || echo unknown)"; do
    [ -n "$_h" ] || continue
    if [[ "$_h" =~ ^(.*)-(master|worker)-[0-9]+$ ]]; then
      RUN_TAG=${BASH_REMATCH[1]}
      break
    fi
  done
  [ -z "$RUN_TAG" ] && RUN_TAG="3axisws_$(date +%Y%m%d-%H%M%S)"
fi

# ── 调试日志目录（共享盘，pod 回收不丢；r$RANK 前缀区分两个 pod）──────────────
LOG_DIR="$REPO_DIR/recipe/agent_evolver/runs/logs/$RUN_TAG"
mkdir -p "$LOG_DIR"

# ── 每 pod GPU 数：RESOURCE_GPU 优先，否则 nvidia-smi ────────────────────────
if [ -n "$RESOURCE_GPU" ] && [ "$RESOURCE_GPU" -gt 0 ]; then
  N_GPU=$RESOURCE_GPU
else
  N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
fi
if ! [[ "$N_GPU" =~ ^[0-9]+$ ]] || [ "$N_GPU" -lt 1 ]; then
  echo "[3xcws][R$RANK] ERROR: cannot detect GPU count (RESOURCE_GPU=$RESOURCE_GPU, nvidia-smi gave '$N_GPU')" >&2
  exit 1
fi

# ── 解析 meta-agent 后端：preset 模型路径存在（多机 pod）→ 本地 vLLM serve，
#    并让出 META_TP 张卡给 meta（agent vLLM 减到 AGENT_N）；不存在（本地）→ 回退
#    anthropic 网关。必须早于 APIS 的 csv_urls 拼串（那里用 AGENT_N）。────────
meta_gpus() {  # 逗号拼接的 CUDA_VISIBLE_DEVICES = 最后 META_TP 张卡
  local out="" i
  for (( i = N_GPU - META_TP; i < N_GPU; i++ )); do
    out="${out}${out:+,}$i"
  done
  echo "$out"
}
resolve_meta() {
  if [ -d "$META_MODEL_PATH" ]; then
    META_USE_PRESET=1
    META_MODEL="$META_MODEL_NAME"
    META_API_BASE="http://127.0.0.1:$META_PORT/v1"
    AGENT_N=$((N_GPU - META_TP))
    META_GPUS=$(meta_gpus)
    echo "[3xcws][R$RANK] meta: preset $META_MODEL_PATH → :$META_PORT (TP=$META_TP on GPUs $META_GPUS; agent vLLM ×$AGENT_N)"
  else
    META_USE_PRESET=0
    META_MODEL="${EVOLVER_META_MODEL:-anthropic/volcengine_maas/deepseek-v4-pro}"
    META_API_BASE="${EVOLVER_META_API_BASE:-${ANTHROPIC_BASE_URL:-https://api.llm.mioffice.cn/anthropic}}"
    AGENT_N=$N_GPU
    META_GPUS=""
    echo "[3xcws][R$RANK] meta: preset $META_MODEL_PATH absent → anthropic gateway (agent vLLM ×$AGENT_N)"
  fi
}
resolve_meta
if [ "$AGENT_N" -lt 1 ]; then
  echo "[3xcws][R$RANK] ERROR: META_TP=$META_TP >= N_GPU=$N_GPU leaves no GPU for agent vLLM (AGENT_N=$AGENT_N)" >&2
  exit 1
fi

# ── 每 rank 的 arm 序列 ───────────────────────────────────────────────────────
# RANK 0：llm 先、enforce 后（串行）；RANK 1：prior。两节点都要 kev。
case "$RANK" in
  0) MY_ARMS=(llm enforce);;
  1) MY_ARMS=(prior);;
  *) echo "[3xcws][R$RANK] ERROR: RANK=$RANK not in 0..1 — submit as a 2-pod job" >&2; exit 1;;
esac
NEED_KEV=1                              # R0 给 enforce、R1 给 prior

# ── 端点串 ────────────────────────────────────────────────────────────────────
csv_urls() {  # $1 = port base ; $2 = count ; $3 = suffix
  local base="$1" n="$2" suffix="$3" out="" i
  for (( i = 0; i < n; i++ )); do
    out="${out}${out:+,}http://127.0.0.1:$((base + i))${suffix}"
  done
  echo "$out"
}
APIS=$(csv_urls "$PORT_BASE" "$AGENT_N" "/v1")
WS_ENV_URL="http://127.0.0.1:$WS_ENV_PORT"
CONC=$AGENT_N

# 集群节点缺 python3.10-dev → 指 gcc 去共享盘头文件；无 CUDA 工具链 → 绕开 FlashInfer JIT
if [ -d "$PYINC_BUNDLE" ] && [ ! -d /usr/include/python3.10 ]; then
  export CPATH="$PYINC_BUNDLE${CPATH:+:$CPATH}"
fi
export VLLM_USE_FLASHINFER_SAMPLER=0

VLLM_FLAGS="--host 127.0.0.1 --served-model-name $MODEL_NAME \
--gpu-memory-utilization $GPU_MEM_UTIL --max-model-len $MAX_MODEL_LEN \
--max-num-seqs $MAX_NUM_SEQS --tensor-parallel-size 1 --dtype bfloat16 \
--gdn-prefill-backend triton --no-enable-log-requests --max-logprobs 40"

# ── helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[3xcws][R$RANK] $(date '+%H:%M:%S') $*"; }
die()  { echo "[3xcws][R$RANK] ERROR: $*" >&2; exit 1; }
vllm_up() { curl -s -m 3 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }
env_up()  { curl -s -m 3 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }
kev_up()  { curl -s -m 3 "http://127.0.0.1:$KEV_PORT/v1/models" >/dev/null 2>&1; }
track_pid() { echo "$1" >> "$PIDFILE"; }

# ── 每 pod preflight（fail fast on 已知集群镜像缺口）─────────────────────────
preflight() {
  local _fail=0
  if [ ! -f /usr/include/python3.10/Python.h ] && [ ! -f "$PYINC_BUNDLE/Python.h" ]; then
    echo "!! missing Python.h (vLLM model-inspection compile needs it)"; _fail=1
  fi
  command -v gcc >/dev/null 2>&1 || { echo "!! missing gcc"; _fail=1; }
  [ -x "$VLLM_BIN" ] || { echo "!! missing vLLM bin: $VLLM_BIN"; _fail=1; }
  [ -d "$MODEL_PATH" ] || { echo "!! missing model dir: $MODEL_PATH"; _fail=1; }
  if [ "$META_USE_PRESET" = "1" ]; then
    [ -f "$META_MODEL_PATH/config.json" ] || { echo "!! missing meta model config.json: $META_MODEL_PATH"; _fail=1; }
  fi
  if [ "$NEED_KEV" = "1" ]; then
    [ -x "$KEV_REPO/.venv/bin/python" ] || { echo "!! missing kev venv python: $KEV_REPO/.venv/bin/python"; _fail=1; }
    [ -d "$KEV_RUN" ] || { echo "!! missing kev run dir: $KEV_RUN"; _fail=1; }
  fi
  # WebShop env server 依赖
  [ -x "$WS_PY" ] || { echo "!! missing webshop python: $WS_PY"; _fail=1; }
  [ -d "$WS_JAVA_HOME" ] || { echo "!! missing webshop JAVA_HOME: $WS_JAVA_HOME"; _fail=1; }
  [ -f "$WS_DATA/items_shuffle.json" ] || { echo "!! missing webshop items_shuffle.json: $WS_DATA"; _fail=1; }
  [ -d "$WS_ROOT/web_agent_site" ] || { echo "!! missing webshop root: $WS_ROOT"; _fail=1; }
  [ "$_fail" -ne 0 ] && die "preflight failed"
  log "preflight OK (Python.h + gcc + vLLM + model + kev + webshop${META_USE_PRESET:+ + meta})  N_GPU=$N_GPU agent_vLLM=$AGENT_N  arms=[${MY_ARMS[*]}]"
}

# ── vLLM serve（每卡一个，agent 模型；preset meta 时只起 AGENT_N 个，留最后 META_TP 张给 meta）──
start_vllm() {
  local gpu port logf pid
  log "starting $AGENT_N vLLM servers on :$PORT_BASE..$((PORT_BASE + AGENT_N - 1))"
  for (( gpu = 0; gpu < AGENT_N; gpu++ )); do
    port=$((PORT_BASE + gpu))
    logf="$LOG_DIR/vllm_${MODEL_NAME}_r${RANK}_${port}.log"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$VLLM_BIN" serve "$MODEL_PATH" $VLLM_FLAGS --port "$port" \
      >"$logf" 2>&1 </dev/null &
    pid=$!
    track_pid "$pid"
    log "vLLM :$port pid=$pid (GPU $gpu)"
  done
}

# ── meta-agent vLLM serve（preset 模型，专用端口，最后 META_TP 张卡）────────────
start_meta_vllm() {
  [ "$META_USE_PRESET" = "1" ] || return 0
  if vllm_up "$META_PORT"; then log "meta vLLM :$META_PORT already up"; return 0; fi
  local logf="$LOG_DIR/vllm_meta_${META_MODEL_NAME}_r${RANK}_${META_PORT}.log"
  log "launching meta vLLM :$META_PORT (model $META_MODEL_NAME, TP=$META_TP on GPUs $META_GPUS)"
  CUDA_VISIBLE_DEVICES="$META_GPUS" nohup "$VLLM_BIN" serve "$META_MODEL_PATH" \
    --host 127.0.0.1 --served-model-name "$META_MODEL_NAME" --port "$META_PORT" \
    --gpu-memory-utilization "$META_GPU_MEM_UTIL" --max-model-len "$META_MAX_MODEL_LEN" \
    --max-num-seqs 16 --tensor-parallel-size "$META_TP" --dtype bfloat16 \
    --no-enable-log-requests >"$logf" 2>&1 </dev/null &
  track_pid "$!"
  log "meta vLLM :$META_PORT pid=$!"
}

# ── kev 的 base 模型 symlink 进 HF cache（pod 的 HF cache 是空的，否则 AutoTokenizer
#    按 hub id 解析时去连 huggingface.co 失败）─────────────────────────────────
kev_hf_cache() {
  local cache_dir="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${KEV_BASE_HUB_ID//\//--}"
  if [ ! -d "$KEV_BASE_DIR" ]; then
    log "WARN: kev base dir missing: $KEV_BASE_DIR (kev will fail to load tokenizer)"
    return 1
  fi
  mkdir -p "$cache_dir/snapshots" "$cache_dir/refs"
  ln -sfn "$KEV_BASE_DIR" "$cache_dir/snapshots/$KEV_BASE_REV"
  printf '%s' "$KEV_BASE_REV" > "$cache_dir/refs/main"
  log "kev HF cache symlink → $KEV_BASE_DIR (rev ${KEV_BASE_REV:0:10}…)"
}

# ── kev sidecar（两节点都要：R0 enforce / R1 prior）──────────────────────────
start_kev() {
  [ "$NEED_KEV" = "1" ] || return 0
  if kev_up; then log "kev :$KEV_PORT already up"; return 0; fi
  kev_hf_cache || return 1
  log "launching kev :$KEV_PORT on GPU 0"
  (
    cd "$KEV_REPO" || exit 1
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 KEV_PREFIX_CACHE=0 CUDA_VISIBLE_DEVICES=0 \
      nohup .venv/bin/python -m kev.serve --run "$KEV_RUN" --port "$KEV_PORT" \
      > "$LOG_DIR/kev_serve_r${RANK}_${KEV_PORT}.log" 2>&1 </dev/null &
    track_pid "$!"
  )
  log "kev :$KEV_PORT launched"
}

# ── WebShop env server（每 pod 一个，官方 full 集，精确 PID）──────────────────
start_ws_env() {
  if env_up "$WS_ENV_PORT"; then log "WebShop :$WS_ENV_PORT already up"; return 0; fi
  log "launching WebShop env :$WS_ENV_PORT (official full set, seed 233 — loads ~1-2 min)"
  JAVA_HOME="$WS_JAVA_HOME" PATH="$WS_JAVA_HOME/bin:$PATH" \
    WEBSHOP_ROOT="$WS_ROOT" \
    WEBSHOP_NUM_PRODUCTS=full WEBSHOP_HUMAN_GOALS=1 WEBSHOP_SEED=233 \
    WEBSHOP_FILE_PATH="$WS_DATA/items_shuffle.json" \
    WEBSHOP_ATTR_PATH="$WS_DATA/items_ins_v2.json" \
    nohup "$WS_PY" "$REPO_DIR/benchmarks/webshop/env_server.py" --host 127.0.0.1 --port "$WS_ENV_PORT" \
    > "$LOG_DIR/webshop_env_r${RANK}_${WS_ENV_PORT}.log" 2>&1 </dev/null &
  pid=$!
  track_pid "$pid"
  log "WebShop :$WS_ENV_PORT pid=$pid"
}

# ── 等本 pod 本地服务就绪（vLLM N + kev + WebShop env 1 + meta）───────────────
wait_local() {
  local waited=0 v e k m ok g
  log "waiting for local infra ..."
  while true; do
    v=0; for (( g = 0; g < AGENT_N; g++ )); do vllm_up $((PORT_BASE + g)) && v=$((v+1)); done
    e=0; env_up "$WS_ENV_PORT" && e=1
    k=1; [ "$NEED_KEV" = "1" ] && ! kev_up && k=0
    m=1; [ "$META_USE_PRESET" = "1" ] && ! vllm_up "$META_PORT" && m=0
    ok=1; [ "$v" = "$AGENT_N" ] || ok=0; [ "$e" = "1" ] || ok=0; [ "$k" = "1" ] || ok=0; [ "$m" = "1" ] || ok=0
    [ "$ok" = "1" ] && { log "local infra up (vLLM $v/$AGENT_N, webshop env $e/1, kev $k/1, meta $m/1)"; return 0; }
    waited=$((waited + 5))
    [ "$waited" -ge 900 ] && die "local infra never ready (vLLM $v/$AGENT_N webshop $e/1 kev $k/1 meta $m/1 — see $LOG_DIR)"
    sleep 5
  done
}

# ── arm / probe / summary ─────────────────────────────────────────────────────
arm_flags() {  # $1 = arm name → 输出该 arm 的 decision 参数（无空格，word-split 安全）
  case "$1" in
    llm)     : ;;
    prior)   echo "--decision-backend kev --decision-mode prior   --decision-base-url http://127.0.0.1:$KEV_PORT" ;;
    enforce) echo "--decision-backend kev --decision-mode enforce --decision-base-url http://127.0.0.1:$KEV_PORT" ;;
  esac
}

run_arm() {  # $1 = arm name（held-in = 官方 train 切片 --start 1500）
  local arm="$1"
  log "=== ARM START ${RUN_TAG}_${arm}_heldin (train goals $START_TRAIN..) ==="
  "$PY" -m recipe.agent_evolver.run webshop \
    --num-rounds "$ROUNDS" --num-tasks "$NUM_TASKS" --start "$START_TRAIN" --seed "$SEED" \
    --agent-api-bases "$APIS" --env-urls "$WS_ENV_URL" --concurrency "$CONC" \
    --meta-model "$META_MODEL" --meta-api-base "$META_API_BASE" \
    --run-tag "${RUN_TAG}_${arm}_heldin" $(arm_flags "$arm")
  log "=== ARM DONE ${RUN_TAG}_${arm}_heldin (exit $?) ==="
}

best_config() {  # $1 = comparison.json -> R<n>/config.yaml（或空）
  [ -f "$1" ] || return 0
  "$PY" - "$1" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
best = max(d["rounds"], key=lambda r: r["mean_reward"])
print(f"R{best['round']}/config.yaml")
PYEOF
}

probe() {  # $1 = run_tag ; $2 = base_config yaml（held-out = 官方 test 切片 --start 0）
  local tag="$1" cfg="$2"
  log "=== HELDOUT PROBE $tag <- $cfg (test goals $START_TEST..) ==="
  "$PY" -m recipe.agent_evolver.run webshop \
    --num-rounds 1 --num-tasks "$NUM_TASKS" --start "$START_TEST" --seed "$SEED" \
    --agent-api-bases "$APIS" --env-urls "$WS_ENV_URL" --concurrency "$CONC" \
    --meta-model "$META_MODEL" --meta-api-base "$META_API_BASE" \
    --run-tag "$tag" --base-config "$cfg"
  log "=== HELDOUT PROBE DONE $tag (exit $?) ==="
}

summary() {
  local t cmp
  log "=== FINAL SUMMARY ==="
  for t in "${RUN_TAG}_llm_heldin" "${RUN_TAG}_prior_heldin" "${RUN_TAG}_enforce_heldin" \
           "${RUN_TAG}_baseline_heldout" "${RUN_TAG}_llmbest_heldout" "${RUN_TAG}_priorbest_heldout" "${RUN_TAG}_enforcebest_heldout"; do
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
  log "=== THREE-AXIS WEBSHOP CLOUDML COMPLETE ==="
}

# ── teardown：只杀本 pod 记录的精确 PID（绝不 pattern-kill）────────────────────
teardown() {
  if [ ! -f "$PIDFILE" ]; then
    log "no pidfile $PIDFILE — nothing to kill"
    return 0
  fi
  log "killing this pod's tracked PIDs from $PIDFILE"
  local pid
  while read -r pid; do
    [ -n "$pid" ] || continue
    kill -TERM "$pid" 2>/dev/null || true
    kill -TERM -- "-$pid" 2>/dev/null || true
  done < "$PIDFILE"
  sleep 5
  while read -r pid; do
    [ -n "$pid" ] || continue
    kill -KILL "$pid" 2>/dev/null || true
    kill -KILL -- "-$pid" 2>/dev/null || true
  done < "$PIDFILE"
  rm -f "$PIDFILE"
  log "teardown done"
}

# ── main ──────────────────────────────────────────────────────────────────────
preflight
start_vllm
start_meta_vllm
start_kev
start_ws_env
wait_local

cd "$REPO_DIR" || die "no repo: $REPO_DIR"

# 跑本 pod 的 arm 序列（R0: llm→enforce 串行；R1: prior），每个 arm 写一个 done 屏障
for arm in "${MY_ARMS[@]}"; do
  run_arm "$arm"
  touch "$DONE_DIR/.done_three_axis_ws.$RUN_TAG.$arm"
  log "arm $arm done → $DONE_DIR/.done_three_axis_ws.$RUN_TAG.$arm"
done
log "my arms done: [${MY_ARMS[*]}]"

# RANK 0：等 prior（RANK 1）后补 held-out 探针 + summary
if [ "$RANK" = "0" ]; then
  log "joining on prior done-file in $DONE_DIR"
  _waited=0
  while [ ! -f "$DONE_DIR/.done_three_axis_ws.$RUN_TAG.prior" ]; do
    _waited=$((_waited + 60))
    [ "$_waited" -ge 86400 ] && die "timed out waiting for prior arm (24h)"
    [ $((_waited % 600)) -eq 0 ] && log "still joining (waited ${_waited}s)"
    sleep 60
  done
  log "all arms done — running 4 held-out probes (test goals, same :$WS_ENV_PORT)"

  probe "${RUN_TAG}_baseline_heldout" "$RUNS/${RUN_TAG}_llm_heldin/R0/config.yaml"

  LLM_BEST=$(best_config "$RUNS/${RUN_TAG}_llm_heldin/comparison.json")
  [ -n "$LLM_BEST" ] && probe "${RUN_TAG}_llmbest_heldout" "$RUNS/${RUN_TAG}_llm_heldin/$LLM_BEST"

  PRIOR_BEST=$(best_config "$RUNS/${RUN_TAG}_prior_heldin/comparison.json")
  [ -n "$PRIOR_BEST" ] && probe "${RUN_TAG}_priorbest_heldout" "$RUNS/${RUN_TAG}_prior_heldin/$PRIOR_BEST"

  ENFORCE_BEST=$(best_config "$RUNS/${RUN_TAG}_enforce_heldin/comparison.json")
  [ -n "$ENFORCE_BEST" ] && probe "${RUN_TAG}_enforcebest_heldout" "$RUNS/${RUN_TAG}_enforce_heldin/$ENFORCE_BEST"

  summary
fi

teardown
log "pod finished (RANK $RANK, arms [${MY_ARMS[*]}])"
