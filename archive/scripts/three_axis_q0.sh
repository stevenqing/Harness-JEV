#!/usr/bin/env bash
# Three-axis kev comparison (temp=0, deterministic rollout).
#
#   llm arm     : evolve on heldin, 6 rounds, meta-LLM does all diagnosis
#   prior arm   : --decision-backend kev --decision-mode prior  (soft prior)
#   enforce arm : --decision-backend kev --decision-mode enforce (hard lever + intent retrocheck)
#
#   held-out probes (single rollout, temp=0): baseline + best config of each arm
#
# Benchmark semantics:
#   alfworld — run.py --split heldin/heldout（独立 env 池）：
#     - held-in  env :18082-18085（eval_in_distribution）
#     - held-out env :18086-18089（eval_out_of_distribution，Phase 4 拉起）
#     - 不传 --env-urls，靠 run.py 按 split 的默认端口。
#   webshop — 单一官方 human goal 池（12087 goals, seed 233, 单 env :18090）：
#     - held-in  = 官方 train 切片 goals 1500..1563 → --start 1500
#     - held-out = 官方 test 切片 goals 0..63     → --start 0
#     - 无需 held-out env 阶段（同一 :18090，--start 0 复用）。
#
# Prereqs (infra from start_three_axis_infra.sh <bench>):
#   - vLLM Qwen3.5-4B on 8200-8203
#   - kev sidecar on 8090  (required by prior/enforce arms)
#   - benchmark env server(s) as above
#
# Launch detached so it survives session exit:
#   setsid nohup bash three_axis_q0.sh <alfworld|webshop> >> /tmp/three_axis_q0.log 2>&1 &
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
KEV_URL="http://127.0.0.1:8090"
export EVOLVER_AGENT_TEMPERATURE=0

log() { echo "[3x/$BENCH $(date +%H:%M:%S)] $*"; }

kev_up() {
  curl -s --max-time 3 "$KEV_URL/" >/dev/null 2>&1
}

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
log "three_axis_q0 start ($BENCH, temp=0, 6 rounds, 4 cards, llm vs prior vs enforce + heldout)"

# ── Phase 1: llm arm (no decision backend) ──────────────────────────────────
run_arm ${PREFIX}_llm_heldin

# ── Phase 2: prior arm (soft) ───────────────────────────────────────────────
if kev_up; then
  run_arm ${PREFIX}_prior_heldin --decision-backend kev --decision-mode prior
else
  log "kev DOWN at $KEV_URL — skipping prior arm"
fi

# ── Phase 3: enforce arm (hard) ─────────────────────────────────────────────
if kev_up; then
  run_arm ${PREFIX}_enforce_heldin --decision-backend kev --decision-mode enforce
else
  log "kev DOWN at $KEV_URL — skipping enforce arm"
fi

# ── Phase 4: held-out env servers (alfworld only) ───────────────────────────
if [ "$BENCH" = alfworld ]; then
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
fi

# ── Phase 5: held-out probes ────────────────────────────────────────────────
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
log "=== THREE-AXIS COMPLETE ($BENCH) ==="
