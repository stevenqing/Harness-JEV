#!/usr/bin/env bash
# WebShop three-axis kev comparison (temp=0, deterministic rollout).
#
#   llm arm     : evolve on held-in (train goals), 6 rounds, meta-LLM diagnoses alone
#   prior arm   : --decision-backend kev --decision-mode prior   (soft prior)
#   enforce arm : --decision-backend kev --decision-mode enforce (hard lever + intent retrocheck)
#
#   held-out probes (single rollout, temp=0): baseline + best config of each arm
#
# WebShop 语义（与 ALFWorld 不同）：
#   - run.py 里 webshop 永远 split="test"（忽略 --split），靠 --start 切 goal 池。
#   - 单个人类 goal 池（12087 goals，env seed 233，单 env server :18090）。
#   - held-in  = official train goals 1500..1563 → --start 1500
#   - held-out = official test  goals 0..63     → --start 0
#   - 无需 held-out env 阶段（同一 :18090，--start 0 复用）。
#
# Prereqs (infra from start_webshop_three_axis_infra.sh):
#   - vLLM Qwen3.5-4B on 8200-8203
#   - WebShop env server on 18090  (single, official full set, 12087 goals)
#   - kev sidecar on 8090  (required by prior/enforce arms)
#
# Launch detached so it survives session exit:
#   setsid nohup bash three_axis_webshop_q0.sh >> /tmp/three_axis_webshop_q0.log 2>&1 &
set -uo pipefail

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$REPO/.venv/bin/python"
RUNS="$REPO/recipe/agent_evolver/runs/evolve"
APIS="http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1"
WS_ENV_URL="http://127.0.0.1:18090"
KEV_URL="http://127.0.0.1:8090"
export EVOLVER_AGENT_TEMPERATURE=0

log() { echo "[ws3x $(date +%H:%M:%S)] $*"; }

kev_up() {
  curl -s --max-time 3 "$KEV_URL/" >/dev/null 2>&1
}

run_arm() {  # $1 = run_tag ; remaining args passed through
  local tag="$1"; shift
  log "=== ARM START $tag ==="
  "$PY" -m recipe.agent_evolver.run webshop --num-rounds 6 --num-tasks 64 \
    --start 1500 --seed 0 --env-urls "$WS_ENV_URL" \
    --run-tag "$tag" --agent-api-bases "$APIS" "$@"
  log "=== ARM DONE $tag (exit $?) ==="
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

probe() {  # $1 = run_tag , $2 = base_config yaml
  local tag="$1" cfg="$2"
  log "=== HELDOUT PROBE $tag <- $cfg ==="
  "$PY" -m recipe.agent_evolver.run webshop --num-rounds 1 --num-tasks 64 \
    --start 0 --seed 0 --env-urls "$WS_ENV_URL" \
    --run-tag "$tag" --agent-api-bases "$APIS" --base-config "$cfg"
  log "=== HELDOUT PROBE DONE $tag (exit $?) ==="
}

cd "$REPO" || { echo "no repo" >&2; exit 1; }
log "three_axis_webshop_q0 start (temp=0, 6 rounds, 4 cards, llm vs prior vs enforce + heldout)"

# ── Phase 1: llm arm (no decision backend) ──────────────────────────────────
run_arm ws3x_llm_heldin

# ── Phase 2: prior arm (soft) ───────────────────────────────────────────────
if kev_up; then
  run_arm ws3x_prior_heldin --decision-backend kev --decision-mode prior
else
  log "kev DOWN at $KEV_URL — skipping prior arm"
fi

# ── Phase 3: enforce arm (hard) ─────────────────────────────────────────────
if kev_up; then
  run_arm ws3x_enforce_heldin --decision-backend kev --decision-mode enforce
else
  log "kev DOWN at $KEV_URL — skipping enforce arm"
fi

# ── Phase 4: held-out probes (same :18090, --start 0) ───────────────────────
probe ws3x_baseline_heldout "$RUNS/ws3x_llm_heldin/R0/config.yaml"

LLM_BEST=$(best_config "$RUNS/ws3x_llm_heldin/comparison.json")
[ -n "$LLM_BEST" ] && probe ws3x_llmbest_heldout "$RUNS/ws3x_llm_heldin/$LLM_BEST"

PRIOR_BEST=$(best_config "$RUNS/ws3x_prior_heldin/comparison.json")
[ -n "$PRIOR_BEST" ] && probe ws3x_priorbest_heldout "$RUNS/ws3x_prior_heldin/$PRIOR_BEST"

ENFORCE_BEST=$(best_config "$RUNS/ws3x_enforce_heldin/comparison.json")
[ -n "$ENFORCE_BEST" ] && probe ws3x_enforcebest_heldout "$RUNS/ws3x_enforce_heldin/$ENFORCE_BEST"

# ── Final summary ───────────────────────────────────────────────────────────
log "=== FINAL SUMMARY ==="
for t in ws3x_llm_heldin ws3x_prior_heldin ws3x_enforce_heldin \
         ws3x_baseline_heldout ws3x_llmbest_heldout ws3x_priorbest_heldout ws3x_enforcebest_heldout; do
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
log "=== WEBSHOP THREE-AXIS COMPLETE ==="
