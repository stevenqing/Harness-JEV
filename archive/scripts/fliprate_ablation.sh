#!/usr/bin/env bash
# §3 代码真跑验证：task flip-rate 归因 vs meta predicted_affected 归因（run 内 A/B）。
#
# 两臂都是 kev prior（incumbent, --decision-backend kev --decision-mode prior），
# 唯一变量 = journal back-fill 的归因来源：
#   meta arm  : --task-fliprate-path ""   → 归因用 meta 的 predicted_affected（现状）
#   rate arm  : 默认                       → 归因用 task 历史翻转率 top-k（§3.1）
#
# 直接指标：journal.jsonl 的 gating_attribution.flipped 对真实 flip 的命中率；
# 间接指标：comparison.json 的 heldin pass rate 趋势。3 轮（R0 baseline → R1 首次
# back-fill → R2 meta 读到 fliprate 归因后 evolve），最快看方向。
#
# Prereqs（start_three_axis_infra.sh alfworld 已起）：
#   vLLM 8200-8203 / kev :8090 / ALF env 18082-18089 + 18102-18113 / meta env。
#
# Launch detached:  setsid nohup bash fliprate_ablation.sh >> /tmp/flipab.log 2>&1 &
set -uo pipefail

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$REPO/.venv/bin/python"
RUNS="$REPO/recipe/agent_evolver/runs/evolve"
KEV_URL="http://127.0.0.1:8090"
export EVOLVER_AGENT_TEMPERATURE=0

PREFIX=flipab

# 8 workers/arm = 2 cards x 4 concurrent entries (vLLM batches the 4 per card).
META_APIS="http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1"
RATE_APIS="http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1"

META_ENV="http://127.0.0.1:18082,http://127.0.0.1:18083,http://127.0.0.1:18084,http://127.0.0.1:18085,http://127.0.0.1:18102,http://127.0.0.1:18103,http://127.0.0.1:18104,http://127.0.0.1:18105"
RATE_ENV="http://127.0.0.1:18106,http://127.0.0.1:18107,http://127.0.0.1:18108,http://127.0.0.1:18109,http://127.0.0.1:18110,http://127.0.0.1:18111,http://127.0.0.1:18112,http://127.0.0.1:18113"

log() { echo "[flipab $(date +%H:%M:%S)] $*"; }

kev_up() { curl -s --max-time 3 "$KEV_URL/v1/models" >/dev/null 2>&1; }

run_arm() {  # $1 = run_tag, $2 = api_bases, $3 = env_urls ; remaining args passed through
  local tag="$1" apis="$2" envs="$3"; shift 3
  log "=== ARM START $tag ==="
  "$PY" -m recipe.agent_evolver.run alfworld --split heldin --num-rounds 3 --num-tasks 64 \
    --run-tag "$tag" --agent-api-bases "$apis" --env-urls "$envs" --concurrency 8 "$@"
  log "=== ARM DONE $tag (exit $?) ==="
}

cd "$REPO" || { echo "no repo" >&2; exit 1; }
log "fliprate_ablation start (temp=0, 3 rounds, 2 arms PARALLEL 8-way each)"

if ! kev_up; then
  log "kev DOWN at $KEV_URL — aborting (both arms need kev prior)"
  exit 1
fi
log "kev UP"

run_arm ${PREFIX}_meta_heldin "$META_APIS" "$META_ENV" \
  --decision-backend kev --decision-mode prior --task-fliprate-path "" \
  > /tmp/flipab_meta.log 2>&1 &
META_PID=$!

run_arm ${PREFIX}_rate_heldin "$RATE_APIS" "$RATE_ENV" \
  --decision-backend kev --decision-mode prior \
  > /tmp/flipab_rate.log 2>&1 &
RATE_PID=$!

log "launched: meta arm pid=$META_PID, rate arm pid=$RATE_PID"
wait "$META_PID"; META_EXIT=$?
wait "$RATE_PID"; RATE_EXIT=$?
log "arms finished: meta exit=$META_EXIT, rate exit=$RATE_EXIT"

# ── Final summary ───────────────────────────────────────────────────────────
log "=== FINAL SUMMARY ==="
for t in ${PREFIX}_meta_heldin ${PREFIX}_rate_heldin; do
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
    log "--- $t (missing comparison.json) ---"
  fi
done
log "=== fliprate_ablation DONE ==="
