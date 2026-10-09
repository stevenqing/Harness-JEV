#!/usr/bin/env bash
# Re-run ONLY the prior arm (crashed earlier on a transient WebShop env
# ConnectTimeout) + its held-out probe. Infra must already be up.
#
# Uses the same tag ws3x_prior_heldin (overwrites the partial R0-R2 run).
# /step now has connect-only retry (agents.py _post_retry connect_only=True),
# so the transient ConnectTimeout that killed the first attempt won't crash it.
set -uo pipefail

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$REPO/.venv/bin/python"
RUNS="$REPO/recipe/agent_evolver/runs/evolve"
APIS="http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1"
WS_ENV_URL="http://127.0.0.1:18090"
export EVOLVER_AGENT_TEMPERATURE=0

log() { echo "[ws3x-prior $(date +%H:%M:%S)] $*"; }

best_config() {
  [ -f "$1" ] || return 0
  "$PY" - "$1" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
best = max(d["rounds"], key=lambda r: r["mean_reward"])
print(f"R{best['round']}/config.yaml")
PYEOF
}

cd "$REPO" || { echo "no repo" >&2; exit 1; }
log "prior re-run start"

log "=== ARM START ws3x_prior_heldin ==="
"$PY" -m recipe.agent_evolver.run webshop --num-rounds 6 --num-tasks 64 \
  --start 1500 --seed 0 --env-urls "$WS_ENV_URL" \
  --run-tag ws3x_prior_heldin --agent-api-bases "$APIS" \
  --decision-backend kev --decision-mode prior
log "=== ARM DONE ws3x_prior_heldin (exit $?) ==="

PRIOR_BEST=$(best_config "$RUNS/ws3x_prior_heldin/comparison.json")
if [ -n "$PRIOR_BEST" ]; then
  log "=== HELDOUT PROBE ws3x_priorbest_heldout <- $PRIOR_BEST ==="
  "$PY" -m recipe.agent_evolver.run webshop --num-rounds 1 --num-tasks 64 \
    --start 0 --seed 0 --env-urls "$WS_ENV_URL" \
    --run-tag ws3x_priorbest_heldout --agent-api-bases "$APIS" \
    --base-config "$RUNS/ws3x_prior_heldin/$PRIOR_BEST"
  log "=== HELDOUT PROBE DONE ws3x_priorbest_heldout (exit $?) ==="
else
  log "no comparison.json for prior — probe skipped"
fi

log "=== PRIOR RE-RUN COMPLETE ==="
