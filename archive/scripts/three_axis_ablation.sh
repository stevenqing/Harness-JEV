#!/usr/bin/env bash
# Ablation on the NEW noul-vector prior (lens + meta-induced lever + kev noul).
# Three arms, each heldin 6 rounds + heldout probe, temp=0 deterministic rollout.
#
#   A lens-only    : --decision-prior-source none   (drop the meta-induced lever)
#   B fixed-pred   : --decision-prior-source fixed  (static hand-written lever set)
#   C meta-scorer  : --decision-backend meta        (meta-LLM scores the noul vector)
#
# Incumbent anchors already run: full prior (ov3xn_prior: 0.797 heldin / 0.844
# heldout) and llm (ov3xn_llm: 0.484 / 0.656).  This script adds A/B/C.
#
# Wave 1: A (cards 0/1) + B (cards 2/3) in PARALLEL, 4-way each (1:1 worker:env).
#         Then their heldout probes in parallel.
# Wave 2: C (all 4 cards, 8-way) then its heldout probe.
#
# Launch detached:  setsid nohup bash three_axis_ablation.sh >> /tmp/three_axis_ablation.log 2>&1 &
set -uo pipefail

BENCH="${1:-alfworld}"
PREFIX=ov3xab

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$REPO/.venv/bin/python"
RUNS="$REPO/recipe/agent_evolver/runs/evolve"
export EVOLVER_AGENT_TEMPERATURE=0

# 4-way = 2 cards x 2 concurrent entries; 8-way = 4 cards x 2 concurrent entries.
A_APIS="http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1"
B_APIS="http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1"
C_APIS="http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1,http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1"

A_ENV="http://127.0.0.1:18082,http://127.0.0.1:18083,http://127.0.0.1:18084,http://127.0.0.1:18085"
B_ENV="http://127.0.0.1:18106,http://127.0.0.1:18107,http://127.0.0.1:18108,http://127.0.0.1:18109"
C_ENV="http://127.0.0.1:18102,http://127.0.0.1:18103,http://127.0.0.1:18104,http://127.0.0.1:18105,http://127.0.0.1:18110,http://127.0.0.1:18111,http://127.0.0.1:18112,http://127.0.0.1:18113"

HELDOUT_ENV_A="http://127.0.0.1:18086,http://127.0.0.1:18087,http://127.0.0.1:18088,http://127.0.0.1:18089"
HELDOUT_ENV_B="http://127.0.0.1:18114,http://127.0.0.1:18115,http://127.0.0.1:18116,http://127.0.0.1:18117"
# C heldout runs 8-way → needs 8 env servers (1 worker : 1 env; tatsu is NOT
# thread-safe, 2 workers/env crashes with IndexError: pop from empty list).
HELDOUT_ENV_C="http://127.0.0.1:18086,http://127.0.0.1:18087,http://127.0.0.1:18088,http://127.0.0.1:18089,http://127.0.0.1:18114,http://127.0.0.1:18115,http://127.0.0.1:18116,http://127.0.0.1:18117"

log() { echo "[ablation/$BENCH $(date +%H:%M:%S)] $*"; }

run_arm() {  # $1=tag $2=apis $3=envs $4=concurrency ; rest passed through
  local tag="$1" apis="$2" envs="$3" conc="$4"; shift 4
  log "=== ARM START $tag (conc=$conc) ==="
  "$PY" -m recipe.agent_evolver.run alfworld --split heldin --num-rounds 6 --num-tasks 64 \
    --run-tag "$tag" --agent-api-bases "$apis" --env-urls "$envs" --concurrency "$conc" "$@"
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

probe() {  # $1=tag $2=cfg $3=apis $4=envs $5=concurrency
  local tag="$1" cfg="$2" apis="$3" envs="$4" conc="$5"
  log "=== HELDOUT PROBE $tag <- $cfg ==="
  "$PY" -m recipe.agent_evolver.run alfworld --split heldout --num-rounds 1 --num-tasks 64 \
    --run-tag "$tag" --agent-api-bases "$apis" --env-urls "$envs" --concurrency "$conc" --base-config "$cfg"
  log "=== HELDOUT PROBE DONE $tag (exit $?) ==="
}

summary() {
  local t
  for t in "$@"; do
    local cmp="$RUNS/$t/comparison.json"
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
}

cd "$REPO" || { echo "no repo" >&2; exit 1; }
log "three_axis_ablation start ($BENCH, 3 arms + heldout)"

# ── Wave 1: A lens-only + B fixed-pred in PARALLEL ──────────────────────────
run_arm ${PREFIX}_lensonly_heldin "$A_APIS" "$A_ENV" 4 \
  --decision-backend kev --decision-mode prior --decision-prior-source none \
  > /tmp/three_axis_ablation_lensonly.log 2>&1 &
A_PID=$!
run_arm ${PREFIX}_fixedpred_heldin "$B_APIS" "$B_ENV" 4 \
  --decision-backend kev --decision-mode prior --decision-prior-source fixed \
  > /tmp/three_axis_ablation_fixedpred.log 2>&1 &
B_PID=$!
log "wave1 launched: A=$A_PID B=$B_PID"
wait "$A_PID"; wait "$B_PID"
log "wave1 arms done"

# ── Wave 1b: heldout probes for A + B in PARALLEL ───────────────────────────
A_BEST=$(best_config "$RUNS/${PREFIX}_lensonly_heldin/comparison.json")
B_BEST=$(best_config "$RUNS/${PREFIX}_fixedpred_heldin/comparison.json")
[ -n "$A_BEST" ] && probe ${PREFIX}_lensonly_heldout "$RUNS/${PREFIX}_lensonly_heldin/$A_BEST" \
  "$A_APIS" "$HELDOUT_ENV_A" 4 > /tmp/three_axis_ablation_lensonly_heldout.log 2>&1 &
[ -n "$B_BEST" ] && probe ${PREFIX}_fixedpred_heldout "$RUNS/${PREFIX}_fixedpred_heldin/$B_BEST" \
  "$B_APIS" "$HELDOUT_ENV_B" 4 > /tmp/three_axis_ablation_fixedpred_heldout.log 2>&1 &
wait
log "wave1b heldout probes done"

# ── Wave 2: C meta-scorer (all 4 cards, 8-way) ─────────────────────────────
run_arm ${PREFIX}_metascorer_heldin "$C_APIS" "$C_ENV" 8 \
  --decision-backend meta --decision-mode prior \
  > /tmp/three_axis_ablation_metascorer.log 2>&1
log "wave2 arm C done"

C_BEST=$(best_config "$RUNS/${PREFIX}_metascorer_heldin/comparison.json")
[ -n "$C_BEST" ] && probe ${PREFIX}_metascorer_heldout "$RUNS/${PREFIX}_metascorer_heldin/$C_BEST" \
  "$C_APIS" "$HELDOUT_ENV_C" 8 > /tmp/three_axis_ablation_metascorer_heldout.log 2>&1
log "wave2 heldout probe done"

log "=== FINAL SUMMARY ==="
summary ${PREFIX}_lensonly_heldin ${PREFIX}_fixedpred_heldin ${PREFIX}_metascorer_heldin \
        ${PREFIX}_lensonly_heldout ${PREFIX}_fixedpred_heldout ${PREFIX}_metascorer_heldout
log "=== three_axis_ablation DONE ==="
