#!/usr/bin/env bash
# Control experiment: constant-prior (structure-only) arm.
#
# Same as arm A (lens-only: --decision-prior-source none) but every prior
# decision is overwritten with a FIXED p(true) after scoring.  This keeps the
# *structure* of the prior (the SOUL step-3 machine-rendered table + aggregate
# row + "System-1 trajectory priors" header) while erasing the *values*, to
# separate "does the prior structure help" from "do the base-rate values help".
#
# Anchors (all temp=0, heldout single-round probe):
#   llm (no prior at all)      0.656
#   lens-only (real kev values) 0.875
#   → constant-prior = lens-only structure + flat 0.5 values.
#     ≈0.875 ⇒ values irrelevant, structure does the work
#     ≈0.656 ⇒ base-rate values matter, structure alone is inert
#
# 4-way (cards 0/1), mirroring arm A exactly so it's within-run comparable.
# heldin env 18082-85, heldout env 18086-89 (all free once arm C finishes).
#
# Launch detached:  setsid nohup bash constant_prior_control.sh 0.5 >> /tmp/constant_prior_control.log 2>&1 &
set -uo pipefail

BENCH="${1:-alfworld}"
CONST="${2:-0.5}"
PREFIX=ov3xcp

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$REPO/.venv/bin/python"
RUNS="$REPO/recipe/agent_evolver/runs/evolve"
export EVOLVER_AGENT_TEMPERATURE=0

APIS="http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1"
ENV="http://127.0.0.1:18082,http://127.0.0.1:18083,http://127.0.0.1:18084,http://127.0.0.1:18085"
HELDOUT_ENV="http://127.0.0.1:18086,http://127.0.0.1:18087,http://127.0.0.1:18088,http://127.0.0.1:18089"

log() { echo "[constant-prior/$BENCH $(date +%H:%M:%S)] $*"; }

cd "$REPO" || { echo "no repo" >&2; exit 1; }
log "constant-prior control start (const=$CONST)"

TAG="${PREFIX}_${CONST}_heldin"
log "=== ARM $TAG (lens-only + constant=$CONST) ==="
"$PY" -m recipe.agent_evolver.run alfworld --split heldin --num-rounds 6 --num-tasks 64 \
  --run-tag "$TAG" --agent-api-bases "$APIS" --env-urls "$ENV" --concurrency 4 \
  --decision-backend kev --decision-mode prior --decision-prior-source none \
  --decision-prior-constant "$CONST"
log "=== ARM DONE $TAG (exit $?) ==="

CMP="$RUNS/$TAG/comparison.json"
if [ -f "$CMP" ]; then
  BEST=$("$PY" - "$CMP" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
best = max(d["rounds"], key=lambda r: r["mean_reward"])
print(f"R{best['round']}/config.yaml")
PYEOF
)
  if [ -n "$BEST" ]; then
    HTAG="${PREFIX}_${CONST}_heldout"
    log "=== HELDOUT PROBE $HTAG <- $BEST ==="
    "$PY" -m recipe.agent_evolver.run alfworld --split heldout --num-rounds 1 --num-tasks 64 \
      --run-tag "$HTAG" --agent-api-bases "$APIS" --env-urls "$HELDOUT_ENV" --concurrency 4 \
      --base-config "$RUNS/$TAG/$BEST"
    log "=== HELDOUT PROBE DONE $HTAG (exit $?) ==="
  fi
fi

log "=== constant-prior control DONE ==="
