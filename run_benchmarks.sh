#!/usr/bin/env bash
# Run the Harness-JEV text-protocol evolver on any of the 4 benchmarks, bringing
# up the required infra first (idempotent — skips anything already healthy).
#
#   usage: run_benchmarks.sh <alfworld|webshop|textcraft|sciworld> [extra run.py args...]
#
# ── Secrets (NEVER hardcoded — read from the environment only) ──────────────
#   ANTHROPIC_API_KEY    meta-agent (deepseek-v4-pro) key; needed when --num-rounds > 1
#   EVOLVER_META_MODEL   optional override (default anthropic/volcengine_maas/deepseek-v4-pro)
#   EVOLVER_META_API_BASE optional override (default https://api.llm.mioffice.cn/anthropic)
#
# ── Infra started here (delegates to benchmarks/start_jev_infra.sh) ─────────
#   vLLM Qwen3.5-4B x4             :8200-8203  (GPU 0..3)
#   ALFWorld heldin x4 / heldout x4 :18082-85 / :18086-89
#   WebShop official x1            :18090
#   ScienceWorld pool x1           :18091
#   TextCraft                      none (in-process env)
#
#   Skip the infra pass with SKIP_INFRA=1 (assume everything already up).
#
# Examples:
#   bash run_benchmarks.sh textcraft
#   bash run_benchmarks.sh alfworld --num-rounds 3 --num-tasks 64
#   bash run_benchmarks.sh sciworld --split heldout
#   bash run_benchmarks.sh webshop --decision-backend kev --decision-mode prior
#
# Run all four (detached, one after another):
#   for b in textcraft alfworld webshop sciworld; do
#     setsid nohup bash run_benchmarks.sh "$b" >> "/tmp/run_$b.log" 2>&1 &
#   done
set -uo pipefail

BENCH="${1:?usage: run_benchmarks.sh <alfworld|webshop|textcraft|sciworld> [args...]}"
shift

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$REPO/.venv/bin/python"
cd "$REPO" || { echo "no repo" >&2; exit 1; }

# ── bring up infra (idempotent) ─────────────────────────────────────────────
if [ "${SKIP_INFRA:-0}" != "1" ]; then
  echo "[run_benchmarks] ensuring infra is up (idempotent)..."
  bash "$REPO/benchmarks/start_jev_infra.sh"
fi

APIS="${EVOLVER_AGENT_API_BASES:-http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1}"

case "$BENCH" in
  alfworld)  SPLIT=heldin ;;
  webshop)   SPLIT=heldin ;;    # --split ignored by run.py (forced "test" internally); held-in/out = --start 0/1500
  textcraft) SPLIT=heldin ;;
  sciworld)  SPLIT=heldin ;;
  *) echo "unknown benchmark '$BENCH' (want alfworld|webshop|textcraft|sciworld)" >&2; exit 2 ;;
esac

# ── wait for the benchmark's env server to answer /health (textcraft: none) ──
wait_env() {  # $1 = port ; $2 = label
  local p="$1" label="$2" i
  for i in $(seq 1 60); do
    curl -s --max-time 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "[run_benchmarks] WARN: $label :$p not healthy after ~120s — proceeding anyway" >&2
}
case "$BENCH" in
  alfworld)  for p in 18082 18083 18084 18085; do wait_env "$p" "ALF heldin"; done ;;
  webshop)   wait_env 18090 "WebShop" ;;
  sciworld)  wait_env 18091 "ScienceWorld" ;;
esac

exec "$PY" -m recipe.agent_evolver.run "$BENCH" \
  --split "$SPLIT" \
  --num-tasks 64 \
  --num-rounds 3 \
  --agent-api-bases "$APIS" \
  "$@"
