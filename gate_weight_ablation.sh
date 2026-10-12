#!/usr/bin/env bash
# §3.1 续：fliprate 归因接进 gate 的分数 A/B（run 内 2-arm）。
#
# 上一轮 fliprate_ablation 已证 fliprate top-k 的「诊断 precision」远胜 meta
# predicted_affected（0.545 vs 0.309），但那次归因只进 journal back-fill、没接
# 进 gate，score 差被 meta 随机性污染。这一步把归因接进 gate 决策：
#
#   唯一变量 = --gate-task-weight：
#     uniform arm : --gate-task-weight uniform   → gate 用 plain mean_reward（现状 baseline）
#     fliprate arm: --gate-task-weight fliprate  → gate 用 fliprate 加权 reward（_weighted_fliprate_reward）
#
# 两臂都 kev prior（incumbent）、都默认 --task-fliprate-path（fliprate 表已加载，
# 归因两臂一致），故唯一差异 = gate 的 accept/reject 决策如何给 reward 加权。3 轮
# （R0 baseline → R1/R2 两次 gate 决策）。
#
# Prereqs（已起）：
#   vLLM 8200-8203 / kev :8090 / ALF heldin env 18082-85 + 18102-13（16 个，8/臂）
#   meta env（ANTHROPIC_API_KEY 已 export）
#
# Launch detached:  setsid nohup bash gate_weight_ablation.sh >> /tmp/gwl.log 2>&1 &
set -uo pipefail

REPO=/mnt/llmshared-ssd-hd/shishuqing/Harness-JEV
PY="$REPO/.venv/bin/python"
RUNS="$REPO/recipe/agent_evolver/runs/evolve"
KEV_URL="http://127.0.0.1:8090"
export EVOLVER_AGENT_TEMPERATURE=0

PREFIX=gwl

# 8 workers/arm = 2 cards x 4 concurrent entries（vLLM 每卡批 4）。两臂用不重叠的卡。
UNIFORM_APIS="http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1"
FLIPRATE_APIS="http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1,http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1"

UNIFORM_ENV="http://127.0.0.1:18082,http://127.0.0.1:18083,http://127.0.0.1:18084,http://127.0.0.1:18085,http://127.0.0.1:18102,http://127.0.0.1:18103,http://127.0.0.1:18104,http://127.0.0.1:18105"
FLIPRATE_ENV="http://127.0.0.1:18106,http://127.0.0.1:18107,http://127.0.0.1:18108,http://127.0.0.1:18109,http://127.0.0.1:18110,http://127.0.0.1:18111,http://127.0.0.1:18112,http://127.0.0.1:18113"

log() { echo "[gwl $(date +%H:%M:%S)] $*"; }

kev_up() { curl -s --max-time 3 "$KEV_URL/v1/models" >/dev/null 2>&1; }
env_up()  { curl -s --max-time 3 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }

run_arm() {  # $1 = run_tag, $2 = api_bases, $3 = env_urls ; remaining args passed through
  local tag="$1" apis="$2" envs="$3"; shift 3
  log "=== ARM START $tag ==="
  "$PY" -m recipe.agent_evolver.run alfworld --split heldin --num-rounds 3 --num-tasks 64 \
    --run-tag "$tag" --agent-api-bases "$apis" --env-urls "$envs" --concurrency 8 "$@"
  log "=== ARM DONE $tag (exit $?) ==="
}

cd "$REPO" || { echo "no repo" >&2; exit 1; }
log "gate_weight_ablation start (temp=0, 3 rounds, 2 arms PARALLEL 8-way each)"

if ! kev_up; then
  log "kev DOWN at $KEV_URL — aborting (both arms need kev prior)"
  exit 1
fi
log "kev UP"

for p in 18082 18083 18084 18085 18102 18103 18104 18105 18106 18107 18108 18109 18110 18111 18112 18113; do
  env_up "$p" || log "WARN: ALF heldin :$p DOWN"
done

run_arm ${PREFIX}_uniform_heldin "$UNIFORM_APIS" "$UNIFORM_ENV" \
  --decision-backend kev --decision-mode prior --gate-task-weight uniform \
  > /tmp/gwl_uniform.log 2>&1 &
UNIFORM_PID=$!

run_arm ${PREFIX}_fliprate_heldin "$FLIPRATE_APIS" "$FLIPRATE_ENV" \
  --decision-backend kev --decision-mode prior --gate-task-weight fliprate \
  > /tmp/gwl_fliprate.log 2>&1 &
FLIPRATE_PID=$!

log "launched: uniform arm pid=$UNIFORM_PID, fliprate arm pid=$FLIPRATE_PID"
wait "$UNIFORM_PID"; UNIFORM_EXIT=$?
wait "$FLIPRATE_PID"; FLIPRATE_EXIT=$?
log "arms finished: uniform exit=$UNIFORM_EXIT, fliprate exit=$FLIPRATE_EXIT"

# ── Final summary ───────────────────────────────────────────────────────────
log "=== FINAL SUMMARY ==="
for t in ${PREFIX}_uniform_heldin ${PREFIX}_fliprate_heldin; do
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
log "=== gate_weight_ablation DONE ==="
