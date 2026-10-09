#!/usr/bin/env bash
# A3 — three-arm × 3-seed visibility comparison (S1 decoding, temp=0 meta).
#
#   Arms:  llm   (no decision backend — meta-LLM diagnoses alone)
#          kev   (--decision-backend kev --decision-mode prior  → three-axis soft prior)
#          rule  (--decision-backend rule → deterministic A2 rule-tagger prior, same brief channel)
#
#   Seeds: 1234 / 2345 / 3456  (different ALFWorld game orderings per split)
#   NO enforce mode.  S1 decoding = thinking on + max_tokens 4096 (baseline default).
#
#   Per seed:  3 held-in evolutions (6 rounds × 64 tasks) → 4 held-out probes
#              (baseline + best config of each arm), single rollout each.
#
# Launch detached so it survives session exit:
#   setsid nohup bash a3_three_arm_3seed.sh >> /tmp/a3_three_arm.log 2>&1 &
set -uo pipefail

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$REPO/.venv/bin/python"
RUNS="$REPO/recipe/agent_evolver/runs/evolve"
APIS="http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1"
KEV_URL="http://127.0.0.1:8090"
export EVOLVER_AGENT_TEMPERATURE=0

SEEDS=(1234 2345 3456)

log() { echo "[A3 $(date +%H:%M:%S)] $*"; }

kev_up() { curl -s --max-time 3 "$KEV_URL/" >/dev/null 2>&1; }

heldin_done() {  # $1 = run_tag → 0 if comparison.json has ≥6 rounds
  local cmp="$RUNS/$1/comparison.json"
  [ -f "$cmp" ] || return 1
  "$PY" - "$cmp" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if len(d.get("rounds", [])) >= 6 else 1)
PYEOF
}

run_heldin() {  # $1 = arm , $2 = seed
  local arm="$1" seed="$2"
  local tag="a3_${arm}_s${seed}_heldin"
  if heldin_done "$tag"; then
    log "HELDIN SKIP (already complete): $tag"
    return 0
  fi
  log "=== HELDIN START $tag (arm=$arm seed=$seed) ==="
  local extra=()
  case "$arm" in
    kev)  extra=(--decision-backend kev --decision-mode prior) ;;
    rule) extra=(--decision-backend rule --decision-mode prior) ;;
    llm)  extra=() ;;
    *) log "unknown arm $arm" >&2; return 1 ;;
  esac
  "$PY" -m recipe.agent_evolver.run alfworld --split heldin --num-rounds 6 --num-tasks 64 \
    --seed "$seed" --run-tag "$tag" --agent-api-bases "$APIS" "${extra[@]}"
  log "=== HELDIN DONE $tag (exit $?) ==="
}

best_config() {  # $1 = comparison.json → prints R<n>/config.yaml (or nothing)
  [ -f "$1" ] || return 0
  "$PY" - "$1" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
best = max(d["rounds"], key=lambda r: r["mean_reward"])
print(f"R{best['round']}/config.yaml")
PYEOF
}

probe() {  # $1 = run_tag , $2 = base_config yaml , $3 = seed
  local tag="$1" cfg="$2" seed="$3"
  log "=== HELDOUT PROBE $tag <- $cfg (seed=$seed) ==="
  "$PY" -m recipe.agent_evolver.run alfworld --split heldout --num-rounds 1 --num-tasks 64 \
    --seed "$seed" --run-tag "$tag" --agent-api-bases "$APIS" --base-config "$cfg"
  log "=== HELDOUT DONE $tag (exit $?) ==="
}

cd "$REPO" || { echo "no repo" >&2; exit 1; }
log "A3 start: 3 arms × 3 seeds, held-in 6 rounds / held-out probes, S1 decoding, temp=0"

for seed in "${SEEDS[@]}"; do
  log "########## SEED $seed ##########"

  # ── held-in evolution (3 arms, sequential) ────────────────────────────────
  run_heldin llm  "$seed"
  if kev_up; then
    run_heldin kev  "$seed"
  else
    log "kev DOWN at $KEV_URL — skipping kev arm for seed $seed"
  fi
  run_heldin rule "$seed"

  # ── held-out probes (baseline + best config of each arm) ─────────────────
  BASELINE_CFG="$RUNS/a3_llm_s${seed}_heldin/R0/config.yaml"
  [ -f "$BASELINE_CFG" ] && probe "a3_baseline_s${seed}_heldout" "$BASELINE_CFG" "$seed"

  LLM_BEST=$(best_config "$RUNS/a3_llm_s${seed}_heldin/comparison.json")
  [ -n "$LLM_BEST" ] && probe "a3_llm_s${seed}_heldout" "$RUNS/a3_llm_s${seed}_heldin/$LLM_BEST" "$seed"

  KEV_BEST=$(best_config "$RUNS/a3_kev_s${seed}_heldin/comparison.json")
  [ -n "$KEV_BEST" ] && probe "a3_kev_s${seed}_heldout" "$RUNS/a3_kev_s${seed}_heldin/$KEV_BEST" "$seed"

  RULE_BEST=$(best_config "$RUNS/a3_rule_s${seed}_heldin/comparison.json")
  [ -n "$RULE_BEST" ] && probe "a3_rule_s${seed}_heldout" "$RUNS/a3_rule_s${seed}_heldin/$RULE_BEST" "$seed"
done

log "########## A3 FINAL SUMMARY ##########"
for seed in "${SEEDS[@]}"; do
  for t in a3_llm_s${seed}_heldin a3_kev_s${seed}_heldin a3_rule_s${seed}_heldin \
           a3_baseline_s${seed}_heldout a3_llm_s${seed}_heldout \
           a3_kev_s${seed}_heldout a3_rule_s${seed}_heldout; do
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
log "=== A3 COMPLETE ==="
