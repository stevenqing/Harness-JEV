#!/usr/bin/env bash
# Three-axis comparison on a fixed-manifest benchmark (TextCraft / ScienceWorld),
# temp=0 deterministic rollout.
#
#   llm arm     : evolve on heldin, 6 rounds, meta-LLM does all diagnosis
#   prior arm   : --decision-backend ${JUDGE} --decision-mode prior  (soft prior)
#   enforce arm : --decision-backend ${JUDGE} --decision-mode enforce (hard lever + retrocheck)
#
#   held-out probes (single rollout, temp=0): baseline + best config of each arm
#
# Usage: three_axis_newbench.sh <sciworld|textcraft>
#
# Benchmark semantics (SCIWORLD_SPEC.md / TEXTCRAFT_SPEC.md):
#   - sciworld  : HTTP env server (benchmarks/sciworld/env_server.py) on :18091;
#                 run.py defaults to it, so no --env-urls needed.
#   - textcraft : in-process env (no env server, no --env-urls).
#   - held-in = splits.json["heldin"], held-out = splits.json["heldout"] (64 each);
#     seed is fixed (run.py forces seed=0), no --seed / --start.
#
# Judge backend: kev is ALFWorld-semantics-specific and is NOT expected to
# transfer to these action vocabularies.  Default JUDGE per benchmark:
#   sciworld → meta, textcraft → kev.  Flip with e.g.
#   JUDGE=meta bash three_axis_newbench.sh textcraft
#
# Prereqs: vLLM Qwen3.5-4B on 8200-8203 (+ kev sidecar :8090 if JUDGE=kev).
#
# Launch detached:
#   setsid nohup bash three_axis_newbench.sh sciworld >> /tmp/three_axis_sciworld.log 2>&1 &
set -uo pipefail

BENCH="${1:?usage: three_axis_newbench.sh <sciworld|textcraft>}"
case "$BENCH" in
  sciworld|textcraft) ;;
  *) echo "benchmark must be sciworld or textcraft (got '$BENCH')" >&2; exit 2 ;;
esac

case "$BENCH" in
  sciworld)  DEFAULT_JUDGE=meta; PREFIX=sw3x ;;
  textcraft) DEFAULT_JUDGE=kev;  PREFIX=tc3x ;;
esac
JUDGE="${JUDGE:-$DEFAULT_JUDGE}"
case "$JUDGE" in
  meta|kev) ;;
  *) echo "JUDGE must be meta or kev (got '$JUDGE')" >&2; exit 2 ;;
esac

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$REPO/.venv/bin/python"
RUNS="$REPO/recipe/agent_evolver/runs/evolve"
APIS="http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1"
KEV_URL="http://127.0.0.1:8090"
export EVOLVER_AGENT_TEMPERATURE=0

log() { echo "[3x/$BENCH $(date +%H:%M:%S)] $*"; }

judge_up() {
  if [ "$JUDGE" = kev ]; then
    curl -s --max-time 3 "$KEV_URL/" >/dev/null 2>&1
  else
    return 0   # meta backend needs no sidecar (calls the meta model directly)
  fi
}

run_arm() {  # $1 = run_tag ; remaining args passed through
  local tag="$1"; shift
  log "=== ARM START $tag ==="
  "$PY" -m recipe.agent_evolver.run "$BENCH" --split heldin --num-rounds 6 --num-tasks 64 \
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
  "$PY" -m recipe.agent_evolver.run "$BENCH" --split heldout --num-rounds 1 --num-tasks 64 \
    --run-tag "$tag" --agent-api-bases "$APIS" --base-config "$cfg"
  log "=== HELDOUT PROBE DONE $tag (exit $?) ==="
}

cd "$REPO" || { echo "no repo" >&2; exit 1; }
log "three_axis_$BENCH start (JUDGE=$JUDGE, temp=0, 6 rounds, llm vs prior vs enforce + heldout)"

# ── Phase 1: llm arm (no decision backend) ──────────────────────────────────
run_arm ${PREFIX}_llm_heldin

# ── Phase 2: prior arm (soft) ───────────────────────────────────────────────
if judge_up; then
  run_arm ${PREFIX}_prior_heldin --decision-backend "$JUDGE" --decision-mode prior
else
  log "judge backend ($JUDGE) DOWN at $KEV_URL — skipping prior arm"
fi

# ── Phase 3: enforce arm (hard) ─────────────────────────────────────────────
if judge_up; then
  run_arm ${PREFIX}_enforce_heldin --decision-backend "$JUDGE" --decision-mode enforce
else
  log "judge backend ($JUDGE) DOWN at $KEV_URL — skipping enforce arm"
fi

# ── Phase 4: held-out probes ────────────────────────────────────────────────
probe ${PREFIX}_baseline_heldout "$RUNS/${PREFIX}_llm_heldin/R0/config.yaml"

LLM_BEST=$(best_config "$RUNS/${PREFIX}_llm_heldin/comparison.json")
[ -n "$LLM_BEST" ] && probe ${PREFIX}_llmbest_heldout "$RUNS/${PREFIX}_llm_heldin/$LLM_BEST"

PRIOR_BEST=$(best_config "$RUNS/${PREFIX}_prior_heldin/comparison.json")
[ -n "$PRIOR_BEST" ] && probe ${PREFIX}_priorbest_heldout "$RUNS/${PREFIX}_prior_heldin/$PRIOR_BEST"

ENFORCE_BEST=$(best_config "$RUNS/${PREFIX}_enforce_heldin/comparison.json")
[ -n "$ENFORCE_BEST" ] && probe ${PREFIX}_enforcebest_heldout "$RUNS/${PREFIX}_enforce_heldin/$ENFORCE_BEST"

# ── Final summary ───────────────────────────────────────────────────────────
log "=== FINAL SUMMARY ==="
for t in ${PREFIX}_llm_heldin ${PREFIX}_prior_heldin ${PREFIX}_enforce_heldin \
         ${PREFIX}_baseline_heldout ${PREFIX}_llmbest_heldout ${PREFIX}_priorbest_heldout ${PREFIX}_enforcebest_heldout; do
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
log "=== THREE-AXIS ${BENCH^^} COMPLETE ==="
