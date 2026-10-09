#!/usr/bin/env bash
# meta-self ablation (temp=0, deterministic rollout): the meta-LLM answers the
# three-axis questions itself instead of kev.
#
#   meta arm : --decision-backend meta --decision-mode prior
#              (identical pipeline to the prior arm; the ONLY difference is who
#              does the per-trajectory three-axis judgment — the meta model, not kev)
#
#   held-out probe: replay the meta arm's held-in best config (temp=0).
#
# Benchmark semantics (same as three_axis_q0.sh):
#   alfworld — --split heldin/heldout（独立 env 池，默认端口 18082-18085 / 18086-18089）
#   webshop  — 单一官方 human goal 池（seed 233, :18090）：
#              held-in = train 切片 --start 1500；held-out = test 切片 --start 0
#
# Prereqs (infra from start_three_axis_infra.sh <bench>): vLLM :8200-8203 +
# benchmark env server(s).  kev :8090 is NOT required by this arm.
#
# Launch detached:
#   setsid nohup bash three_axis_meta.sh <alfworld|webshop> >> /tmp/three_axis_meta.log 2>&1 &
set -uo pipefail

BENCH="${1:-${BENCH:-alfworld}}"
case "$BENCH" in
  alfworld|webshop) ;;
  *) echo "usage: $0 [alfworld|webshop]" >&2; exit 2 ;;
esac

if [ "$BENCH" = alfworld ]; then
  PREFIX=ov3x
else
  PREFIX=ws3x
fi

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$REPO/.venv/bin/python"
RUNS="$REPO/recipe/agent_evolver/runs/evolve"
APIS="http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1"
ALF_PY=/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-alfworld/bin/python
ALF_DATA=/mnt/llmshared-ssd-hd/cty/data/alfworld
WS_ENV_URL="http://127.0.0.1:18090"
export EVOLVER_AGENT_TEMPERATURE=0

log() { echo "[3xm/$BENCH $(date +%H:%M:%S)] $*"; }

run_arm() {  # $1 = run_tag ; remaining args passed through
  local tag="$1"; shift
  log "=== ARM START $tag ==="
  if [ "$BENCH" = alfworld ]; then
    "$PY" -m recipe.agent_evolver.run alfworld --split heldin --num-rounds 6 --num-tasks 64 \
      --run-tag "$tag" --agent-api-bases "$APIS" "$@"
  else
    "$PY" -m recipe.agent_evolver.run webshop --num-rounds 6 --num-tasks 64 \
      --start 1500 --seed 0 --env-urls "$WS_ENV_URL" \
      --run-tag "$tag" --agent-api-bases "$APIS" "$@"
  fi
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
  if [ "$BENCH" = alfworld ]; then
    "$PY" -m recipe.agent_evolver.run alfworld --split heldout --num-rounds 1 --num-tasks 64 \
      --run-tag "$tag" --agent-api-bases "$APIS" --base-config "$cfg"
  else
    "$PY" -m recipe.agent_evolver.run webshop --num-rounds 1 --num-tasks 64 \
      --start 0 --seed 0 --env-urls "$WS_ENV_URL" \
      --run-tag "$tag" --agent-api-bases "$APIS" --base-config "$cfg"
  fi
  log "=== HELDOUT PROBE DONE $tag (exit $?) ==="
}

cd "$REPO" || { echo "no repo" >&2; exit 1; }
log "three_axis_meta start ($BENCH, temp=0, 6 rounds, meta-self prior + heldout)"

# ── Phase 1: meta arm (meta-LLM answers the three-axis questions itself) ─────
run_arm ${PREFIX}_meta_heldin --decision-backend meta --decision-mode prior

# ── Phase 2: held-out probe of meta best ─────────────────────────────────────
META_BEST=$(best_config "$RUNS/${PREFIX}_meta_heldin/comparison.json")
[ -n "$META_BEST" ] && probe ${PREFIX}_metabest_heldout "$RUNS/${PREFIX}_meta_heldin/$META_BEST"

# ── Final summary ────────────────────────────────────────────────────────────
log "=== FINAL SUMMARY ==="
for t in ${PREFIX}_meta_heldin ${PREFIX}_metabest_heldout; do
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
log "=== META-SELF ABLATION COMPLETE ($BENCH) ==="
