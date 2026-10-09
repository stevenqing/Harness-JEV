---
name: alfworld-playbook
description: ALFWorld-specific benchmark guidance (ALFRED text-embodied household tasks). Sparse binary reward (won→1.0 else 0.0), driven by the Skill1 `<think>/<action>` ReAct text loop (no native tools). Catalogues ALFWorld failure modes and known levers from the JEV evolution history, mapped to this evolver's editable surfaces (system prompt + guidance + loop params). Use when forming a hypothesis about a find/acquire failure or choosing which lever to turn.
---

# ALFWorld: Benchmark-specific Guidance

## Current Benchmark

- **What**: ALFWorld — the text-grounded ALFRED embodied benchmark. The agent
  completes a household goal (e.g. "put a clean mug on the shelf") inside a
  simulated home by issuing text actions (`go to cabinet 1`, `open cabinet 1`,
  `take mug 1 from cabinet 1`, `put mug 1 in/on microwave 1`).
- **Reward**: binary — `won → 1.0`, otherwise `0.0`. The env itself judges.
- **Splits**: `held-in` (`eval_in_distribution`, 140 games) and `held-out`
  (`eval_out_of_distribution`, 134 games) — **both indexed `game_idx` 0–63** but
  served by different env servers (held-in 18082–85, held-out 18086–89). The
  evolver trains on `held-in`; `held-out` is the generalization probe. Baseline
  Skill1 (Qwen3.5-4B) scores **~0.344 on both**,
  so held-in ≈ held-out is the clean start signal — a lift on held-in that does
  **not** transfer to held-out is prompt overfitting, not generalization.

## Task Structure (this evolver)

The agent is driven by the Skill1 ReAct text loop (see
`recipe/agent_evolver/agents.py:run_alfworld`), NOT native tool-calling. Each
turn the model receives:

1. A **system prompt** (the `prompt:` field — baseline is the bare Qwen default).
2. A **per-turn user prompt** built from the fixed Skill1 template: the goal,
   the recent (obs, action) history, the current observation, and the admissible
   action list, plus a `{skills}` slot filled by **`guidance:`**.
3. The model must reason inside `<think>…</think>` then emit exactly one
   admissible action inside `<action>…</action>`. The action is extracted and
   POSTed to the env; the returned observation/admissible list form the next
   turn.

The loop ends on env `done` (win or fail) or at `max_steps` (50).

## Editable Surfaces (what you may change)

The config YAML is the unit of evolution. Its shape:

```yaml
processors:
  - _target_: harnessx.processors.context.system_prompt.SystemPromptProcessor
    system_builder:
      _target_: file://…/native/prompt.py::StaticSystemPromptBuilder
      prompt: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."  # ← system prompt
  - _target_: file://…/native/text_spec.py::TextEvolverSpecProcessor
    guidance: ""              # ← PRIMARY LEVER: text injected into every user turn (the {skills} slot)
    loop_breaker_k: 3         # secondary: drop an action repeated k steps in a row
    history_mode: raw         # secondary: how past (obs, action) are rendered
    history_n: 2              # secondary: how many recent steps to show
    history_obs_clip: 150     # secondary: obs truncation (only when history_mode=summary)
    tried_action_mask: false  # secondary: hide already-tried actions
```

- **`guidance:`** is the main lever. It is spliced into *every* user turn, so
  it is the natural place for state-tracking rules, find-strategy rules,
  action-sequencing rules. Write it as imperative agent instructions.
- **`prompt:`** is the system message (baseline: bare Qwen default). Rewriting
  it can add global behaviour ("you are an ALFWorld housework agent…"), but the
  per-turn guidance usually carries more signal.
- **`loop_breaker_k`** / **`history_mode`** / **`history_n`** /
  **`history_obs_clip`** / **`tried_action_mask`** are deterministic loop rules
  (see `recipe/agent_evolver/processors.py`). They are known-saturated (below)
  — treat as fine-tune knobs, not the primary lever.
- Do **not** restructure the config: keep the two processors above and their
  `_target_` URIs. Edit only the `prompt:`/`guidance:` strings and the scalar
  params. Adding/removing processors will silently change nothing (the text
  loop only reads these two).

## Analyzing Failures

Frontmatter fields the evolver writes per task (always present):

| Field | Meaning |
| --- | --- |
| `task_id` | game_idx (0–63; same index, different env server per split) |
| `goal` | the goal sentence |
| `exit_reason` | `done` / `max_steps` |
| `eval_reason` | `won` / `failed` (env done, no win) / `incomplete` (hit max_steps) |
| `steps` | env steps taken |
| `invalid_actions` | number of extracted actions NOT in the admissible list (format/enforcement signal) |
| `reward` / `eval_passed` / `eval_score` | `won`→1.0/true/1.0; else 0.0/false/0.0 |
| `final_obs` (failure tier only) | where the episode actually ended |

`eval_passed` is the ground truth. The trajectory body shows each step's raw
model output → extracted action, so you can see the *why* (bad reasoning vs
bad action choice vs malformed `<action>`).

## Common Failure Modes

### A. Find failure — blind search thrash (~60% of failures)

The model cannot locate the target object and burns steps `go to`-ing
containers it has already visited, or examining irrelevant containers.

- **Signal**: `eval_reason=incomplete` or `failed`; `steps` near `max_steps`;
  the body shows repeated `go to <same container>`; `final_obs` still shows a
  room, no target object.
- **Fix direction** (`guidance`): add a rule to track which containers have
  been visited/opened and prefer unvisited containers ("before each `go to`,
  recall the containers you have NOT yet checked and pick from them").

### B. Acquire failure — found it, but didn't take it (~31% of failures)

The object is visible but the model never issues the `take … from …` action,
or takes from a closed container.

- **Signal**: `final_obs` shows the target in view; body shows `go to`/`open`
  then a wrong or missing `take`; `eval_reason=failed`.
- **Fix direction** (`guidance`): "a container must be `open` before you `take`
  from it"; "once you see the target in the observation, `take` it immediately —
  do not `go to` elsewhere first".

### C. Malformed / invented action

The extracted `<action>` is empty, truncated, or not in the admissible list.

- **Signal**: `invalid_actions` > 0; the body shows a `<think>` without a
  closing `<action>`, or an invented action.
- **Fix direction**: reinforce the format in `guidance` ("always end with
  exactly one `<action>…</action>` copied verbatim from the admissible list").
  If `invalid_actions` spikes *across most tasks*, it is usually a generation
  issue (temperature / max_tokens), not a prompt fix.

### D. Premature stop

The model stops emitting `<action>` before the env reports `done`.

- **Signal**: `eval_reason=failed` with small `steps`; last turn has a
  `<think>` but no `<action>`.
- **Fix direction**: `guidance` rule "keep acting until the environment
  reports success/failure — your job is to act every turn, not to declare
  yourself done."

## Known Techniques (from the JEV evolution history)

Discovered on this same text stack; re-discover rather than assume (the exact
prompt wording matters):

| Technique | Lever | Notes |
| --- | --- | --- |
| **Unvisited-container injection** — tell the model which containers it has NOT yet checked | `guidance` | historically the biggest single find-failure fix |
| **Loop-breaker** — forbid re-`go to` already-visited containers | `loop_breaker_k` | historically **net negative** (0.579→0.500): revisit is often necessary; a hard ban hurts |
| **Tried-action mask** — hide actions already issued | `tried_action_mask` | neutral-to-negative; the mask erases the "I already tried this" signal |
| **Positional prior** — name the target's most likely room/container | `guidance` | strong when available; do not hallucinate a location the env never gave |

The recurring lesson: **hard restrictions on the admissible list are already
saturated**; the open lever is a *better state model in the text* (where am I,
what have I visited, what's left to check), not removing options.

## Red Flags — Pipeline Bugs, Not Agent Bugs

Record in `_meta_scratch/NEEDS_FROM_HUMAN.md`; do NOT build an agent-layer fix:

- **`invalid_actions` high across *most* tasks** → the `<action>` extraction is broken or the model is not following the template (a generation/plumbing issue, not a guidance issue).
- **Every task `eval_reason=incomplete` with empty `final_obs`** → env server (18082–18089) down or `/reset`/`/step` erroring.
- **`reward=1.0` but `eval_passed=false`** (or the reverse) → evaluator wiring bug.
- **held-in and held-out both collapse to ~0** → the guidance now breaks the `<action>` format; revert to the previous accepted config.
- **A correct-looking trajectory scoring 0.0** → the env's `won` flag is the only truth; if the goal looks met but `won=false`, it is an env/ground-truth mismatch.

## Pointers

- `recipe/agent_evolver/agents.py` — `run_alfworld` (the ReAct loop), templates, `extract_action`.
- `recipe/agent_evolver/processors.py` — `mask_admissible` / `render_history` (the loop rules).
- `recipe/agent_evolver/spec.py` — `HarnessSpec`, `QWEN_SYSTEM`, `ALF_MAX_STEPS`.
- `recipe/agent_evolver/native/config.py` — baseline config + `spec_from_config`.
