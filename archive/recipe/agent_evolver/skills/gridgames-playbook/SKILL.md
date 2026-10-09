---
name: gridgames-playbook
description: FrozenLake / Sokoban grid-game benchmark guidance. Sparse-ish reward (success→1.0, else a continuous `progress_end` in [0,1]) driven by the R0 loop (one model call per move, two messages — system + user — and a FIXED bracket-direction parser). Catalogues the FL/SK failure modes and known levers, mapped to this evolver's editable surfaces (system prompt + guidance + history window). Use when forming a hypothesis about a FrozenLake/Sokoban failure or choosing which lever to turn.
---

# FrozenLake / Sokoban: Benchmark-specific Guidance

## Current Benchmark

- **What**: two grid-game suites from the `gridgames` bundle (`ta_env.LevelEnv`),
  played by a Qwen3.5 model one move at a time.
  - **FrozenLake (FL)**: walk the player `P` to the goal `G` on an `N×N` ice
    grid without stepping on a hole `H` (fall = lose). Symbols: `P`=player,
    `F`=ice (safe), `H`=hole, `G`=goal.
  - **Sokoban (SK)**: push every box onto a target. Symbols: `#`=wall, `_`=floor,
    `O`=target, `P`=player, `S`=player-on-target, `B`=box, `X`=box-on-target.
- **Reward**: `reward = progress_end` — a continuous `[0,1]` value (how far the
  episode progressed toward the goal); `success` is the binary win flag. The
  evolution gate uses **mean `progress_end`** (continuous), success is reported
  separately.
- **Splits**: each cell (game × tier) is three-way split — `evolve` (256, held-in,
  the training split), `gate` (64, fixed uniform-random floor judge), `test` (128,
  held-out). A cell is "live" when `success − floors.gate.uniform ≥ 0.10`.
- **Tiers**: `L4` (8×8, budget 8), `L8` (budget 16), `L16` (budget 32),
  `L32` (16×16, budget 64). Higher tiers need longer-horizon planning.

## Task Structure (this evolver)

The agent is driven by the R0 loop (see `recipe/agent_evolver/gridgame_evolve.py`),
NOT native tool-calling. Each turn the model receives exactly **two messages**:

1. A **system message**: the game rules + symbol legend + move budget, built from
   `system_prompt` (the `{size}`/`{budget}` placeholders are filled by the
   rollout) with `guidance` appended when non-empty.
2. A **user message**: the current board (one symbol per cell), the moves left,
   and the last `history_window` moves with their results.

The model must reply with **exactly one move in brackets** — e.g. `[up]` — or a
lone direction word. The reply is parsed by `r0.parse_action`, which is **FIXED
for every round and every harness** (spec §5). An unreadable reply is a
`parse_fail` move (it still consumes a move). The loop ends when the env reports
done (win / fell in hole / box stuck) or the move budget is exhausted.

## Editable Surfaces (what you may change)

The config YAML is the unit of evolution. Its shape:

```yaml
processors:
  - _target_: file://…/native/gridgame_spec_processor.py::GridgameSpecProcessor
    system_prompt: "You are playing FrozenLake on a {size}x{size} grid …"  # ← system prompt (KEEP {size}/{budget})
    guidance: ""              # ← PRIMARY LEVER: appended to the system message every turn
    history_window: 3         # secondary: how many recent moves are shown
```

- **`guidance:`** is the main lever. It is spliced into *every* turn's system
  message, so it is the natural place for state-tracking rules, hole-avoidance
  rules, "head toward the goal" rules, box-planning rules. Write it as imperative
  agent instructions.
- **`system_prompt:`** is the game-rules text (baseline = R0's rules + symbol
  legend). Rewriting it can add global behaviour, but it must **keep the
  `{size}`/`{budget}` placeholders** — the rollout substitutes only those two
  tokens with `str.replace`; anything else you write stays verbatim.
- **`history_window`** is how many recent `(action → result)` lines are shown.
  Small values reduce noise; large values give more context but dilute the board.
- Do **not** restructure the config: keep the single `GridgameSpecProcessor` and
  its `_target_` URI. The rollout reads only that one processor.
- The **board renderer and the action parser are frozen** — you cannot change how
  symbols are drawn or how a reply is parsed. Intervene on the *text* instead.

## Analyzing Failures

Frontmatter fields the evolver writes per episode (always present):

| Field | Meaning |
| --- | --- |
| `task_id` | `<level_id>_e<k>` (one file per episode; k = episode 0..3) |
| `goal` | `<game> <tier> level <level_id>` |
| `exit_reason` | `done` / `max_steps` |
| `eval_reason` | `won` / `failed` (env done, no win) / `incomplete` (budget exhausted) |
| `steps` | moves taken (includes `parse_fail` moves) |
| `invalid_actions` | number of `parse_fail` (unreadable) moves |
| `reward` / `eval_passed` / `eval_score` | `progress_end` / `success` / `progress_end` |
| `final_obs` (failure tier only) | the readable end-state board |

`eval_passed` is the ground truth. The trajectory body shows each move's board →
raw model output → extracted action, so you can see the *why* (unreadable reply
vs wrong direction vs no spatial reasoning).

## Common Failure Modes

### A. Degenerate direction collapse (~the dominant mode under frozen decoding)

The model emits the *same* direction almost every turn (`[up]`, or `[down]` for
some models), making zero progress and hitting the move budget.

- **Signal**: `eval_reason=incomplete`; `steps ≈ budget`; the body shows the same
  `[dir]` repeated; `reward≈0`.
- **Important context**: this has been traced to **decoding** (thinking off +
  `max_tokens 32` leaves no room to reason about the board), *not* the prompt.
  R0-style prompt edits alone may not move it. Still try guidance that forces a
  *per-turn* direction decision (e.g. "before replying, state which of the four
  neighbours are holes, then pick a non-hole direction").
- If `parse_fail`/`invalid_actions` spikes across **most** tasks, it is a
  generation issue, not a guidance issue.

### B. Walking into a hole / no hole avoidance

The model can see the board but steps onto `H`.

- **Signal**: `eval_reason=failed`; `steps < budget`; `final_obs` shows the player
  fell in a hole; the body shows a move into an adjacent `H`.
- **Fix direction** (`guidance`): "never step onto `H`; check the four neighbours
  of `P` and move toward the goal through `F` cells".

### C. Unreadable reply (parse_fail)

The reply is empty, truncated, or not a single bracketed direction.

- **Signal**: `invalid_actions` > 0; body shows prose or a multi-option reply.
- **Fix direction** (`guidance`): reinforce "reply with exactly one move in
  brackets, e.g. `[up]`". Note the parser requires all bracketed directions to
  agree — avoid listing example alternatives inside brackets.

### D. No goal-seeking (random walk)

The model moves but never reduces the distance to the goal.

- **Signal**: `steps ≈ budget`, `reward` low, `final_obs` far from `G`.
- **Fix direction** (`guidance`): "locate `G` on the board, then move to reduce
  the Manhattan distance to it while avoiding `H`".

### E. (Sokoban only) Pushing a box into a wall / corner

The model pushes boxes into unrecoverable positions.

- **Signal**: `eval_reason=failed`; `final_obs` shows a box against a wall or in a
  corner off any target.
- **Fix direction** (`guidance`): "do not push a box into a wall, another box, or
  a corner; push boxes toward `O` targets".

## Known Techniques (from the JEV evolution history)

| Technique | Lever | Notes |
| --- | --- | --- |
| **State-tracking guidance** — force the model to name the board state before acting | `guidance` | mirrors the ALFWorld "unvisited-container" win: a better state model in text, not harder restrictions |
| **Longer history window** | `history_window` | helps the model notice it is repeating itself; larger windows add noise |
| **Verbose/reasoning-inducing prompt** | `system_prompt` | historically **negative** here (long intros make models emit prose that truncates before the bracketed action under `max_tokens 32`) — keep prompts minimal |
| **Reinterpreting symbols** | `system_prompt` | the X/B/√ meaning differs across renderers; a prompt that mismatches the board's actual symbols causes parse/plan failures |

The recurring lesson: under the frozen decoding, the model has almost no
inference budget, so the text must do the *state modeling* for it — explicitly
name where the player is, what is adjacent, and what the single best next move is.

## Red Flags — Pipeline Bugs, Not Agent Bugs

Record in `_meta_scratch/NEEDS_FROM_HUMAN.md`; do NOT build an agent-layer fix:

- **`invalid_actions` high across *most* tasks** → the reply is being truncated or
  the model is not following the bracket format (a decoding/plumbing issue).
- **Every task `eval_reason=incomplete` with empty `final_obs`** → the vLLM agent
  server is down or `max_tokens`/thinking flags are misconfigured.
- **`reward=1.0` but `eval_passed=false`** (or the reverse) → evaluator wiring bug.
- **held-in and held-out both collapse to ~0** → the guidance broke the bracket
  format; revert to the previous accepted config.
- **A visually-correct trajectory scoring 0.0** → `env.report()["success"]` is the
  only truth; the env's own judge decides.

## Pointers

- `recipe/agent_evolver/gridgame_evolve.py` — `messages_from_spec`, `run_gridgame_spec` (the R0 loop).
- `recipe/agent_evolver/r0.py` — `render_board` (frozen), `parse_action` (fixed).
- `recipe/agent_evolver/gridgame_spec.py` — `GridgameSpec`, baseline prompts.
- `recipe/agent_evolver/native/gridgame_config.py` — baseline config + `gridgame_spec_from_config`.
