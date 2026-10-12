# TextCraft vendored environment

## Source

- Repo: https://github.com/WooooDyy/AgentGym
- Commit: `3ef9235` (Merge pull request #75 from pedestrian6/patch-1)
- Path: `agentenv-textcraft/agentenv_textcraft/`
- License: MIT (FudanNLP-Agent, 2023) — kept verbatim as `LICENSE`.

## Files vendored (verbatim, unmodified)

- `environment.py` — `TextCraftEnv(gym.Env[str, str])`; `step` / `reset`.
- `crafting_tree.py` — `CraftingTree` (recipe graph, depths, `traverse_recipe_tree`).
- `utils.py` — `ItemTag` / `ItemTagWithCount` / `Recipe` / `ActionFailed` / `item_id_to_str`.
  - *Included beyond the spec's literal "3 things" list because both
    `environment.py` and `crafting_tree.py` `from .utils import ...` — it is a
    hard transitive dependency.*
- `recipes/` — 860 recipe JSON files (the full recipe set).

## NOT vendored (deliberately)

- `server.py`, `launch.py`, `model.py`, `env_wrapper.py`, upstream `__init__.py` —
  these are the AgentGym FastAPI server stack (uvicorn/pydantic). The spec
  requires **no** FastAPI server (it is single-process, `/commands` + `/goal`
  call non-existent functions, and its `reset` mutates global `random`). We
  drive `TextCraftEnv` in-process instead.
- Upstream `__init__.py` is replaced by our bare stub so `vendor` is a package
  without pulling the server deps.

## Changes vs upstream

None to the vendored source. Added only this `VENDOR.md` + a bare `__init__.py`.

Note: this is the **AgentGym** copy, whose `traverse_recipe_tree` copies
`visited` per recursion (`new_visited.copy()`), so it does **not** exhibit the
ADaPT (archiki/ADaPT) bug where the recipe list grows in-place on every reset.

## Known environment behaviours (spec §3.4 — do NOT "fix", surface in dumps)

| behaviour | note |
|---|---|
| exact quantity only | `craft 5 leather using 20 rabbit hide` is rejected; craft one at a time and repeat |
| output count ignored | `craft 9 leather using 4 rabbit hide` crafts 1, no error |
| `get` has no resource limit | any amount allowed |
| `get` a craftable item / tag fails | returns `Could not find X` |
| case-sensitive | leading `Craft`, or a trailing `.`, fails |
| crafting a distractor recipe succeeds | no error, but does not help the goal |

## Dependency

`gymnasium` (+ its `numpy` transitively). `transformers` is NOT required.
