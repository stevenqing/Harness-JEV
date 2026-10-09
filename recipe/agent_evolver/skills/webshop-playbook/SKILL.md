---
name: webshop-playbook
description: WebShop-specific benchmark guidance (text e-commerce agent: search + click to buy the product matching an instruction). Continuous product-match reward in [0,1]; pass threshold reward ≥ 0.999. Driven by the Skill1 `<think>/<action>` ReAct text loop. Catalogues WebShop failure modes and levers, mapped to this evolver's editable surfaces (system prompt + guidance + loop params). Use when forming a hypothesis about a search/click/attribute-matching failure.
---

# WebShop: Benchmark-specific Guidance

## Current Benchmark

- **What**: WebShop — the text e-commerce benchmark (Yao et al.). A customer
  instruction names a product with desired attributes (e.g. "I'm looking for a
  long sofa in brown, and price lower than 1200 dollars"); the agent navigates
  search + product-detail pages with only `search[…]` and `click[…]` actions,
  and must buy the matching product.
- **Reward**: continuous `[0,1]` product-match score; **pass = `reward ≥
  0.999`** (the exact correct product). A near-miss (right type, wrong color)
  scores partial reward.
- **Split**: a single test set (task_id 0–63) served by one env server
  (`18090`). Baseline Skill1 (Qwen3.5-4B) scores **pass ~0.281, mean reward
  ~0.535**.

## Task Structure (this evolver)

The agent is driven by the Skill1 ReAct text loop (see
`recipe/agent_evolver/agents.py:run_webshop`), NOT native tool-calling. Each
turn the model receives:

1. A **system prompt** (the `prompt:` field — baseline is the bare Qwen default).
2. A **per-turn user prompt** built from the fixed Skill1 template: the
   instruction, the recent (obs, action) history, the current page observation,
   and the available `search[…]`/`click[…]` list, plus a `{skills}` slot filled
   by **`guidance:`**.
3. The model reasons inside `<think>…</think>` then emits one action inside
   `<action>…</action>` (`search[query]` or `click[text]`).

The loop ends on env `done` (purchase or give-up) or at `max_steps` (15).

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
  it is the natural place for attribute-extraction rules, search-query
  construction rules, and buy-decision rules. Write it as imperative
  instructions.
- **`prompt:`** is the system message (baseline: bare Qwen default).
- The five scalar params are deterministic loop rules (see
  `recipe/agent_evolver/processors.py`); historically neutral-to-negative for
  WebShop, so treat as fine-tune knobs.
- Do **not** restructure the config: keep the two processors and their
  `_target_` URIs; edit only the `prompt:`/`guidance:` strings and the scalar
  params. Adding/removing processors will silently change nothing (the text
  loop only reads these two).

## Analyzing Failures

Frontmatter fields the evolver writes per task (always present):

| Field | Meaning |
| --- | --- |
| `task_id` | WebShop task id (0–63) |
| `goal` | the customer instruction |
| `exit_reason` | `done` / `max_steps` |
| `eval_reason` | `won` (reward≥0.999) / `failed` (env done, <0.999) / `incomplete` (hit max_steps) |
| `steps` | env steps taken |
| `invalid_actions` | extracted actions not valid for the current page |
| `reward` | float `[0,1]` — the product-match score |
| `eval_passed` / `eval_score` | `reward≥0.999`→true; else false / reward |
| `final_obs` (failure tier only) | final page + whether purchase completed |

**`reward` is the actionable signal.** 0.7–0.9 = bought a *near* product (right
type, wrong attribute); read the body to see which attribute was misread.
0.0 = gross mismatch or gave up without buying.

## Common Failure Modes

### A. Wrong search query / never narrows down

The agent searches a generic term, then clicks the first result without
comparing attributes.

- **Signal**: few `search[…]` calls; the clicked product doesn't match the
  instruction's attributes; partial `reward`.
- **Fix direction** (`guidance`): "extract every attribute (type/color/size/price
  bound) from the instruction FIRST, then `search` a query that includes the
  discriminating attribute, and verify each attribute against the product page
  before `click[Buy Now]`."

### B. Attribute misread at click time

The agent clicks the wrong attribute option (e.g. wrong color) or ignores a
price/attribute constraint.

- **Signal**: `reward` 0.5–0.95; body shows the correct attribute on screen
  that the agent skipped.
- **Fix direction**: "when the instruction names a color/size/price, the clicked
  option text must contain that value; re-read the page before Buy Now."

### C. Buys without enough evidence / premature purchase

The agent clicks `Buy Now` from the first product page without confirming all
attributes match.

- **Signal**: small `steps`, purchase completes, `reward < 0.999`.
- **Fix direction**: "do not `click[Buy Now]` until you have confirmed every
  stated attribute on the product page."

### D. Invalid action / clicks a non-element

The agent emits `search[…]` when no search bar is present, or `click[…]` on
text that isn't clickable.

- **Signal**: `invalid_actions` > 0; page unchanged.
- **Fix direction**: reinforce "only click text that literally appears in the
  available-actions list; only `search[…]` when a search bar is shown."

### E. Give-up / premature stop

The agent stops before buying (thinks it can't find it).

- **Signal**: `eval_reason=failed` or `incomplete`, `reward=0.0`, no purchase.
- **Fix direction**: "search at least a couple of reformulated queries before
  concluding the product is absent; keep acting until the env reports the
  purchase is complete."

## Known Techniques

| Technique | Lever | Notes |
| --- | --- | --- |
| **Instruction→attribute checklist** — first turn enumerates required attributes, then search + verify each | `guidance` | the highest-leverage prompt move on WebShop |
| **Attribute-constrained search** — put the discriminating attribute (color/size/price) in the `search` query | `guidance` | narrows the candidate list before any click |
| **Tried-action mask / loop-breaker** | `tried_action_mask` / `loop_breaker_k` | historically neutral-to-negative; WebShop revisits the product list legitimately |

## Red Flags — Pipeline Bugs, Not Agent Bugs

Record in `_meta_scratch/NEEDS_FROM_HUMAN.md`; do NOT build an agent-layer fix:

- **`invalid_actions` high across *most* tasks** → the `<action>` extraction is broken or the model is not following the template (generation/plumbing, not guidance).
- **Every task `eval_reason=incomplete` with empty `final_obs`** → env server `18090` down or `/create`/`/step` erroring.
- **`reward` reported but `eval_passed` always false** (or the reverse) → threshold/evaluator wiring bug.
- **All tasks pass ~0.281 with mean reward ~0.535 every round and nothing moves** → the guidance/prompt edits are not reaching the rollout (config not re-decoded); check `spec_from_config` reads the evolved fields.
- **A product that visually matches scoring `reward=0.0`** → the env's reward is the only truth; if the purchased product looks correct, it is an env/ground-truth mismatch.

## Pointers

- `recipe/agent_evolver/agents.py` — `run_webshop` (the ReAct loop), templates, `_ws_action_list`, `_ws_is_valid`.
- `recipe/agent_evolver/processors.py` — `mask_admissible` / `render_history` (the loop rules).
- `recipe/agent_evolver/spec.py` — `HarnessSpec`, `QWEN_SYSTEM`, `WS_MAX_STEPS`.
- `recipe/agent_evolver/native/config.py` — baseline config + `spec_from_config`.
