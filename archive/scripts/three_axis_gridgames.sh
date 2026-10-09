#!/usr/bin/env bash
# Gridgames three-axis (llm vs prior vs enforce) on the frozen held-in/held-out
# datasets (16 levels per tier × L4/L8/L16/L32 = 64 levels per file, per game).
#
# Mirrors the WebShop/ALF three-axis: 6 evolve rounds on held-in, then a 1-round
# held-out probe from each arm's best config.  Uses --levels-file (frozen levels,
# --game/--tier/--split ignored for level selection) + --episodes-per-level 1 so
# one "task" = one level = one trajectory (same 64-task granularity as ALF/WS).
#
# Agent = Qwen3.5-4B (vLLM :8200-8203, thinking ON).  Meta = deepseek-v4-pro.
# Decision sidecar (kev) must be up at :8090 for the prior/enforce arms.
set -uo pipefail

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$REPO/.venv/bin/python"
RUNS="$REPO/recipe/agent_evolver/runs/evolve_gridgames"
BUNDLE=/mnt/llmshared-ssd-hd/shishuqing/gridgames
APIS="http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1"

log() { echo "[gg3x $(date +%H:%M:%S)] $*"; }

best_config() {
  [ -f "$1" ] || return 0
  "$PY" - "$1" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
# gridgames gate key = pass_rate (success fraction); tie-break on mean_reward.
best = max(d["rounds"], key=lambda r: (r["pass_rate"], r["mean_reward"]))
print(f"R{best['round']}/config.yaml")
PYEOF
}

run_arm() {
  local game="$1" tag="$2" levels_file="$3"
  shift 3
  log "=== ARM START $tag (levels=$(wc -l < "$levels_file")) ==="
  "$PY" -m recipe.agent_evolver.run_gridgames --game "$game" \
    --levels-file "$levels_file" --num-rounds 6 --episodes-per-level 1 \
    --run-tag "$tag" --agent-api-bases "$APIS" "$@"
  log "=== ARM DONE $tag (exit $?) ==="
}

run_probe() {
  local game="$1" tag="$2" base="$3" heldout="$4"
  log "=== HELDOUT PROBE $tag <- $base ==="
  "$PY" -m recipe.agent_evolver.run_gridgames --game "$game" \
    --levels-file "$heldout" --num-rounds 1 --episodes-per-level 1 \
    --run-tag "$tag" --agent-api-bases "$APIS" --base-config "$base"
  log "=== HELDOUT PROBE DONE $tag (exit $?) ==="
}

cd "$REPO" || { echo "no repo" >&2; exit 1; }

for game in frozenlake sokoban; do
  heldin="$BUNDLE/heldin_${game}.jsonl"
  heldout="$BUNDLE/heldout_${game}.jsonl"

  run_arm "$game" "${game}3x_llm_heldin"      "$heldin"
  run_arm "$game" "${game}3x_prior_heldin"    "$heldin" --decision-backend kev --decision-mode prior
  run_arm "$game" "${game}3x_enforce_heldin"  "$heldin" --decision-backend kev --decision-mode enforce

  for arm in llm prior enforce; do
    BEST=$(best_config "$RUNS/${game}3x_${arm}_heldin/comparison.json")
    if [ -n "$BEST" ]; then
      run_probe "$game" "${game}3x_${arm}best_heldout" \
        "$RUNS/${game}3x_${arm}_heldin/$BEST" "$heldout"
    else
      log "no comparison.json for ${game}3x_${arm}_heldin — probe skipped"
    fi
  done
done

log "=== ALL GRIDGAMES 3AXIS COMPLETE ==="
