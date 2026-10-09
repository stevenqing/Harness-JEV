#!/usr/bin/env bash
# Overnight Harness-JEV quality experiment (temp=0, deterministic rollout).
#
#   Phase 1  llm arm : evolve on heldin, 6 rounds, no decision priors
#   Phase 2  kev arm : evolve on heldin, 6 rounds, --decision-backend kev
#   Phase 3  bring up held-out env servers (18086-18089)
#   Phase 4  held-out generalization probes (single rollout, temp=0):
#              baseline config  + llm best config  + kev best config
#
# Launch detached so it survives session exit:
#   setsid nohup bash overnight_q0.sh >> /tmp/overnight_q0.log 2>&1 &
set -uo pipefail

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$REPO/.venv/bin/python"
RUNS="$REPO/recipe/agent_evolver/runs/evolve"
APIS="http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1"
ALF_PY=/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-alfworld/bin/python
ALF_DATA=/mnt/llmshared-ssd-hd/cty/data/alfworld
export EVOLVER_AGENT_TEMPERATURE=0

log() { echo "[ov $(date +%H:%M:%S)] $*"; }

run_arm() {  # $1 = run_tag ; remaining args passed through
  local tag="$1"; shift
  log "=== ARM START $tag ==="
  "$PY" -m recipe.agent_evolver.run alfworld --split heldin --num-rounds 6 --num-tasks 64 \
    --run-tag "$tag" --agent-api-bases "$APIS" "$@"
  log "=== ARM DONE $tag (exit $?) ==="
}

best_config() {  # $1 = comparison.json -> prints R<n>/config.yaml
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
  "$PY" -m recipe.agent_evolver.run alfworld --split heldout --num-rounds 1 --num-tasks 64 \
    --run-tag "$tag" --agent-api-bases "$APIS" --base-config "$cfg"
  log "=== HELDOUT PROBE DONE $tag (exit $?) ==="
}

cd "$REPO" || { echo "no repo" >&2; exit 1; }
log "overnight_q0 start (temp=0, 6 rounds, 4 cards, llm vs kev + heldout)"

# ── Phase 1: llm arm ────────────────────────────────────────────────────────
run_arm ov_q0_llm_heldin

# ── Phase 2: kev arm ────────────────────────────────────────────────────────
run_arm ov_q0_kev_heldin --decision-backend kev

# ── Phase 3: held-out env servers ───────────────────────────────────────────
log "bringing up held-out env 18086-18089"
for p in 18086 18087 18088 18089; do
  if curl -s --max-time 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1; then
    log "held-out :$p already up"; continue
  fi
  ALFWORLD_DATA="$ALF_DATA" ALFWORLD_SPLIT=eval_out_of_distribution \
    setsid nohup "$ALF_PY" "$REPO/benchmarks/alfworld/env_server.py" --port "$p" \
    > "/tmp/alfworld_heldout_$p.log" 2>&1 &
  log "held-out :$p launched"
done
for i in $(seq 1 24); do
  up=0; for p in 18086 18087 18088 18089; do
    curl -s --max-time 3 "http://127.0.0.1:$p/health" >/dev/null 2>&1 && up=$((up+1))
  done
  [ "$up" = 4 ] && { log "held-out env all up"; break; }
  sleep 10
done

# ── Phase 4: held-out probes ────────────────────────────────────────────────
probe ov_q0_baseline_heldout "$RUNS/ov_q0_llm_heldin/R0/config.yaml"

LLM_BEST=$(best_config "$RUNS/ov_q0_llm_heldin/comparison.json")
log "llm best config = $LLM_BEST"
probe ov_q0_llmbest_heldout "$RUNS/ov_q0_llm_heldin/$LLM_BEST"

KEV_BEST=$(best_config "$RUNS/ov_q0_kev_heldin/comparison.json")
log "kev best config = $KEV_BEST"
probe ov_q0_kevbest_heldout "$RUNS/ov_q0_kev_heldin/$KEV_BEST"

# ── Final summary ───────────────────────────────────────────────────────────
log "=== FINAL SUMMARY ==="
for t in ov_q0_llm_heldin ov_q0_kev_heldin ov_q0_baseline_heldout ov_q0_llmbest_heldout ov_q0_kevbest_heldout; do
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
log "=== OVERNIGHT COMPLETE ==="
