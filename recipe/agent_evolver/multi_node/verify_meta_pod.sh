#!/usr/bin/env bash
# =============================================================================
# verify_meta_pod.sh — 在 cloudml/Volcano pod 上验证多机 meta-agent 的三件事
# =============================================================================
#   ① preset 模型路径是否真的挂载了（/preset-models/public/DeepSeek-V4.1-Flash）
#   ② vLLM 能不能 serve 它（单卡 dry-run，端口 8401，不碰真端口 8400）
#   ③ pod 是否够得到 mioffice anthropic 网关（确认网络隔离这个根因还在）
#
# 用法（单独提交一个 GPU pod 跑这条命令即可，不依赖任何已启动的服务）：
#   bash /mnt/llmshared-ssd-hd/shishuqing/Harness-JEV/recipe/agent_evolver/multi_node/verify_meta_pod.sh
#
#   默认单卡（META_TP=1）。若①通过但②报 OOM，用多卡重试：
#   META_TP=2 bash .../verify_meta_pod.sh
#
# 只读探测 + 一次临时 vLLM dry-run（跑完自动杀掉），不改任何共享状态、不碰 agent 端口。
# =============================================================================
set -uo pipefail

VLLM_BIN="/mnt/llmshared-ssd-hd/shishuqing/.venv/bin/vllm"
META_MODEL_PATH="/preset-models/public/DeepSeek-V4.1-Flash"
META_MODEL_NAME="DeepSeek-V4.1-Flash"
META_TP="${META_TP:-1}"
DRY_PORT=8401
DRY_TIMEOUT_S="${DRY_TIMEOUT_S:-300}"

say()  { echo "[verify-meta] $(date '+%H:%M:%S') $*"; }
pass() { echo "  [PASS] $*"; }
fail() { echo "  [FAIL] $*"; }

dry_gpus() {  # "0" / "0,1" / ... = 前 META_TP 张卡
  local i out=""
  for (( i = 0; i < META_TP; i++ )); do out="${out}${out:+,}$i"; done
  echo "$out"
}

ready=0   # ② 的判定结果，避免 set -u 下 ② 被跳过时未绑定

# ── ① preset 模型挂载 ────────────────────────────────────────────────────────
say "================ ① preset 模型挂载 ================"
if [ -f "$META_MODEL_PATH/config.json" ]; then
  pass "config.json 存在: $META_MODEL_PATH/config.json"
  echo "  -- config.json 前 800 字节 --"
  head -c 800 "$META_MODEL_PATH/config.json" 2>/dev/null; echo
else
  fail "config.json 缺失: $META_MODEL_PATH/config.json"
fi
echo "  -- /preset-models/public/ 实际内容 --"
ls -la /preset-models/public/ 2>&1 || echo "  /preset-models/public/ 不存在（挂载没上来）"

# ── ② vLLM dry-run serve ─────────────────────────────────────────────────────
say "================ ② vLLM dry-run serve (TP=$META_TP, GPUs=$(dry_gpus), :$DRY_PORT) ================"
if [ ! -x "$VLLM_BIN" ]; then
  fail "vLLM bin 缺失: $VLLM_BIN"
elif [ ! -f "$META_MODEL_PATH/config.json" ]; then
  echo "  （跳过：① 未通过，先解决挂载）"
else
  logf="/tmp/verify_meta_vllm_${DRY_PORT}.log"
  : > "$logf"
  CUDA_VISIBLE_DEVICES="$(dry_gpus)" "$VLLM_BIN" serve "$META_MODEL_PATH" \
    --host 127.0.0.1 --served-model-name "$META_MODEL_NAME" --port "$DRY_PORT" \
    --gpu-memory-utilization 0.90 --max-model-len 32768 \
    --max-num-seqs 16 --tensor-parallel-size "$META_TP" --dtype bfloat16 \
    --no-enable-log-requests >"$logf" 2>&1 </dev/null &
  vllm_pid=$!
  say "vLLM pid=$vllm_pid, log=$logf, 最多等 ${DRY_TIMEOUT_S}s"

  ready=0
  waited=0
  while [ "$waited" -lt "$DRY_TIMEOUT_S" ]; do
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
      echo "  （vLLM 进程已退出 → 大概率崩溃，看 log）"
      break
    fi
    code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$DRY_PORT/v1/models" 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then ready=1; break; fi
    sleep 5
    waited=$((waited + 5))
  done

  if [ "$ready" = "1" ]; then
    pass "vLLM 能 serve $META_MODEL_NAME (TP=$META_TP)，:$DRY_PORT/v1/models 返回 200"
  else
    fail "vLLM 未能 serve（超时或崩溃）。关键报错行："
    grep -iE 'out of memory|oom|cuda|architecture|not support|error|exception|traceback|valueerror|unknown model' "$logf" 2>/dev/null | tail -20 || true
    echo "  -- log 末尾 15 行 --"
    tail -15 "$logf" 2>/dev/null || true
  fi
  kill "$vllm_pid" 2>/dev/null || true
  pkill -f "vllm serve $META_MODEL_PATH" 2>/dev/null || true
fi

# ── ③ mioffice 网关可达性 ────────────────────────────────────────────────────
say "================ ③ mioffice 网关可达性 ================"
code=$(curl -s -m 8 -o /dev/null -w '%{http_code}' https://api.llm.mioffice.cn/anthropic/v1/models 2>/dev/null || echo "TIMEOUT/UNREACHABLE")
case "$code" in
  200|401|403) pass "mioffice 可达（http=$code；401/403 = 网络通、只是没带 key）" ;;
  *)           fail "mioffice 不可达（http=$code）→ 坐实网络隔离，preset 是唯一解" ;;
esac

# ── 结论速览 ─────────────────────────────────────────────────────────────────
say "================ 结论速览 ================"
cfg_ok=0; vllm_ok=0
[ -f "$META_MODEL_PATH/config.json" ] && cfg_ok=1
[ "$ready" = "1" ] && vllm_ok=1

if [ "$cfg_ok" = "1" ] && [ "$vllm_ok" = "1" ]; then
  say "①② 全过 → 多机可直接跑：run_multinode.sh / three_axis_cloudml.sh 会自动 detect preset 走 :8400"
elif [ "$cfg_ok" = "1" ] && [ "$vllm_ok" = "0" ]; then
  say "① 过、② 挂 → 看上面 log：OOM 就 META_TP=2 重跑本脚本；architecture 报错就换 vLLM 版本/换 meta 模型"
else
  say "① 挂 → /preset-models/public/DeepSeek-V4.1-Flash 没挂上，去平台侧补 mount（或改 META_MODEL_PATH 指向真实路径）"
fi
