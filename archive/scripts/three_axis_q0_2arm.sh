#!/usr/bin/env bash
# Two-arm comparison (temp=0, deterministic rollout) on the NEW meta-induced
# noul-vector prior (decision_prior.py no longer has the abstract 4-way lever).
#
#   llm arm   : evolve on heldin, 6 rounds, meta-LLM does all diagnosis (no prior)
#   prior arm : --decision-backend kev --decision-mode prior
#               (meta induces K binary predicates each round; kev scores a noul vector)
#
#   held-out probes (single rollout, temp=0): baseline + best config of each arm
#
# The enforce arm is intentionally absent: its single-lever binding was retired
# (the lever axis is now a soft noul vector), so the meaningful comparison is
# llm vs prior only.
#
# SPEEDUP vs the old sequential script:
#   - the two arms run in PARALLEL on split cards (llm: GPU0/1, prior: GPU2/3),
#     so each arm's meta-evolve (~13 min/round, the dominant cost) overlaps;
#   - each arm batches 4 concurrent sequences per card (8 workers/arm, vLLM
#     --max-num-seqs 256 absorbs them), ~4x rollout throughput.
#
# Benchmark semantics (alfworld): run.py --split heldin/heldout（独立 env 池）
#   - held-in  env :18082-18085 + 18102-18113 (16 total, 8 per arm)
#   - held-out env :18086-18089 (eval_out_of_distribution)
#   - 不传 --env-urls 时靠 run.py 按 split 的默认端口；这里显式传 8 个 heldin。
#
# Prereqs (infra from start_three_axis_infra.sh alfworld):
#   - vLLM Qwen3.5-4B on 8200-8203
#   - kev sidecar on 8090  (prior arm)
#   - ALFWorld env servers 18082-18089 (heldin/heldout) + 18102-18113 (extra heldin)
#   - meta backend env (ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL) — needed by BOTH
#     arms (llm's evolve + prior's per-round predicate induction)
#
# Launch detached so it survives session exit:
#   setsid nohup bash three_axis_q0_2arm.sh >> /tmp/three_axis_q0_2arm.log 2>&1 &
set -uo pipefail

BENCH="${1:-alfworld}"
case "$BENCH" in
  alfworld|webshop) ;;
  *) echo "usage: $0 [alfworld|webshop]" >&2; exit 2 ;;
esac

if [ "$BENCH" = alfworld ]; then
  PREFIX=ov3xn   # ov3x + "noul" — distinct from the old abstract-lever ov3x runs
else
  PREFIX=ws3xn
fi

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$REPO/.venv/bin/python"
RUNS="$REPO/recipe/agent_evolver/runs/evolve"
ALF_PY=/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-alfworld/bin/python
ALF_DATA=/mnt/llmshared-ssd-hd/cty/data/alfworld
WS_ENV_URL="http://127.0.0.1:18090"
KEV_URL="http://127.0.0.1:8090"
export EVOLVER_AGENT_TEMPERATURE=0

# 8 workers/arm = 2 cards x 4 concurrent entries (vLLM batches the 4 per card).
# llm arm -> GPU0/1 (8200/8201), prior arm -> GPU2/3 (8202/8203); kev stays on
# GPU0 but only the prior arm calls it, and prior's rollout is on GPU2/3.
LLM_APIS="http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1"
PRIOR_APIS="http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1"
ALL_APIS="http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1"

LLM_ENV="http://127.0.0.1:18082,http://127.0.0.1:18083,http://127.0.0.1:18084,http://127.0.0.1:18085,http://127.0.0.1:18102,http://127.0.0.1:18103,http://127.0.0.1:18104,http://127.0.0.1:18105"
PRIOR_ENV="http://127.0.0.1:18106,http://127.0.0.1:18107,http://127.0.0.1:18108,http://127.0.0.1:18109,http://127.0.0.1:18110,http://127.0.0.1:18111,http://127.0.0.1:18112,http://127.0.0.1:18113"

log() { echo "[2arm/$BENCH $(date +%H:%M:%S)] $*"; }

kev_up() {
  curl -s --max-time 3 "$KEV_URL/v1/models" >/dev/null 2>&1
}

run_arm() {  # $1 = run_tag, $2 = api_bases, $3 = env_urls ; remaining args passed through
  local tag="$1" apis="$2" envs="$3"; shift 3
  log "=== ARM START $tag ==="
  if [ "$BENCH" = alfworld ]; then
    "$PY" -m recipe.agent_evolver.run alfworld --split heldin --num-rounds 6 --num-tasks 64 \
      --run-tag "$tag" --agent-api-bases "$apis" --env-urls "$envs" --concurrency 8 "$@"
  else
    "$PY" -m recipe.agent_evolver.run webshop --num-rounds 6 --num-tasks 64 \
      --start 1500 --seed 0 --env-urls "$WS_ENV_URL" \
      --run-tag "$tag" --agent-api-bases "$apis" --concurrency 8 "$@"
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
      --run-tag "$tag" --agent-api-bases "$ALL_APIS" --concurrency 4 --base-config "$cfg"
  else
    "$PY" -m recipe.agent_evolver.run webshop --num-rounds 1 --num-tasks 64 \
      --start 0 --seed 0 --env-urls "$WS_ENV_URL" \
      --run-tag "$tag" --agent-api-bases "$ALL_APIS" --concurrency 4 --base-config "$cfg"
  fi
  log "=== HELDOUT PROBE DONE $tag (exit $?) ==="
}

cd "$REPO" || { echo "no repo" >&2; exit 1; }
log "three_axis_q0_2arm start ($BENCH, temp=0, 6 rounds, 2 arms PARALLEL 8-way each + heldout)"

# ── Phase 1+2: run both arms in PARALLEL ──────────────────────────────────────
if ! kev_up; then
  log "kev DOWN at $KEV_URL — aborting (prior arm needs kev)"
  exit 1
fi

run_arm ${PREFIX}_llm_heldin "$LLM_APIS" "$LLM_ENV" > /tmp/three_axis_q0_2arm_llm.log 2>&1 &
LLM_PID=$!
run_arm ${PREFIX}_prior_heldin "$PRIOR_APIS" "$PRIOR_ENV" --decision-backend kev --decision-mode prior \
  > /tmp/three_axis_q0_2arm_prior.log 2>&1 &
PRIOR_PID=$!

log "launched: llm arm pid=$LLM_PID, prior arm pid=$PRIOR_PID"
wait "$LLM_PID"; LLM_EXIT=$?
wait "$PRIOR_PID"; PRIOR_EXIT=$?
log "arms finished: llm exit=$LLM_EXIT, prior exit=$PRIOR_EXIT"

# ── Phase 3: held-out env servers (alfworld only) ────────────────────────────
if [ "$BENCH" = alfworld ]; then
  log "ensuring held-out env 18086-18089"
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
fi

# ── Phase 4: held-out probes ─────────────────────────────────────────────────
probe ${PREFIX}_baseline_heldout "$RUNS/${PREFIX}_llm_heldin/R0/config.yaml"

LLM_BEST=$(best_config "$RUNS/${PREFIX}_llm_heldin/comparison.json")
[ -n "$LLM_BEST" ] && probe ${PREFIX}_llmbest_heldout "$RUNS/${PREFIX}_llm_heldin/$LLM_BEST"

PRIOR_BEST=$(best_config "$RUNS/${PREFIX}_prior_heldin/comparison.json")
[ -n "$PRIOR_BEST" ] && probe ${PREFIX}_priorbest_heldout "$RUNS/${PREFIX}_prior_heldin/$PRIOR_BEST"

# ── Final summary ───────────────────────────────────────────────────────────
log "=== FINAL SUMMARY ==="
for t in ${PREFIX}_llm_heldin ${PREFIX}_prior_heldin \
         ${PREFIX}_baseline_heldout ${PREFIX}_llmbest_heldout ${PREFIX}_priorbest_heldout; do
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
log "=== three_axis_q0_2arm DONE ==="
