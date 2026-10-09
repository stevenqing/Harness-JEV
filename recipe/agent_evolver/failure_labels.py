# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Independent failure labels (visibility spec A1).

A label is ground truth built from the *oracle*, never from the model's text:

- **ALFWorld** — the expert plan's target object and the post-``take`` action,
  read from the task's ``traj_data.json`` (``plan.high_pddl``).  ``find`` = the
  agent never ``take``s the target; ``acquire`` = it ``take``s it but never
  completes the next required expert action; ``other`` = anything else.
- **gridgames** — the four env-state modes ``fell_in_hole`` / ``dead_push`` /
  ``loop`` / ``budget``, recovered by *replaying* the recorded action sequence
  through the bundle's ``ta_env.LevelEnv`` (deterministic, in-process, no model).

These are the reference answers the rule tagger (A2) and the kev reviewer are
both scored against.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

_BUNDLE_DIR = Path(os.environ.get(
    "EVOLVER_GRIDGAMES_DIR", "/mnt/llmshared-ssd-hd/shishuqing/gridgames")).resolve()
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))
import ta_env  # noqa: E402  (bundle LevelEnv)


# ── shared action parsing ─────────────────────────────────────────────────────

# PDDL high-level action name -> the verb the agent uses in its text action.
_PDDL_VERB = {
    "PickupObject": "take",
    "CleanObject": "clean",
    "HeatObject": "heat",
    "CoolObject": "cool",
    "PutObject": "put",
    "SliceObject": "slice",
    "ToggleObject": "use",   # the agent may also say "toggle <lamp>"
    "OpenObject": "open",
    "ExamineObject": "examine",
    "GotoLocation": "go to",
}
_TOGGLE_VERBS = {"use", "toggle"}


def parse_agent_action(action: str) -> tuple[str, str]:
    """Return ``(verb, obj)`` for a lowercased ALFWorld text action.

    ``go to cabinet 1`` -> ``("go to", "cabinet")``; ``take mug 1 from cabinet
    1`` -> ``("take", "mug")``; ``use desklamp 1`` -> ``("use", "desklamp")``;
    ``look`` -> ``("look", "")``.
    """
    a = (action or "").strip().lower()
    if not a:
        return ("", "")
    toks = a.split()
    if toks[0] == "go" and len(toks) >= 3 and toks[1] == "to":
        return ("go to", toks[2])
    if len(toks) == 1:
        return (toks[0], "")
    return (toks[0], toks[1])


def has_loop(actions: list[str]) -> bool:
    """Deterministic loop test: a dominant direction collapse, or a repeated 2-cycle."""
    acts = [a for a in actions if a]
    n = len(acts)
    if n < 4:
        return False
    top, top_n = Counter(acts).most_common(1)[0]
    if top_n >= 3 and top_n / n >= 0.5:
        return True
    for i in range(n - 5):
        a, b = acts[i], acts[i + 1]
        if a == b:
            continue
        rep, j = 1, i + 2
        while j + 1 <= n - 1 and acts[j] == a and acts[j + 1] == b:
            rep += 1
            j += 2
        if rep >= 3:
            return True
    return False


# ── ALFWorld: label from the expert plan (traj_data.json) ─────────────────────

def alf_requirement(task: dict) -> tuple[str | None, str | None, str | None]:
    """Return ``(target, req_verb, req_obj)`` for a traj_data.json dict.

    ``target`` = the object the expert picks up first (``PickupObject.args[0]``,
    already lowercased in ALFWorld's ``high_pddl``).  The requirement is the
    first non-navigation action after that pickup — for a ``ToggleObject`` the
    requirement is ``("use", toggle_target)`` (matching either ``use`` or
    ``toggle``).  Returns ``(None, None, None)`` when the plan has no pickup.
    """
    plan = task.get("plan", {}).get("high_pddl", [])
    acts: list[tuple[str, str]] = []
    for step in plan:
        da = step.get("discrete_action", {})
        name = da.get("action")
        args = da.get("args", [])
        if name:
            acts.append((name, args[0] if args else ""))
    take_i = next((i for i, (n, _) in enumerate(acts) if n == "PickupObject"), None)
    if take_i is None:
        return None, None, None
    target = acts[take_i][1]
    for name, arg in acts[take_i + 1:]:
        if name == "GotoLocation":
            continue
        if name == "ToggleObject":
            return target, "use", arg
        return target, _PDDL_VERB.get(name, name.lower()), arg
    return target, None, None


def alfworld_label(actions: list[str], task: dict) -> str:
    """``find`` / ``acquire`` / ``other`` for a failed ALFWorld trajectory.

    ``actions`` = the extracted action strings in order; ``task`` = the
    traj_data.json dict.  Issuing ``take <target>`` at all (even inadmissible)
    proves the agent located the target, so the find/acquire cut is made on the
    *attempt*, matching the playbook's find (blind search) vs acquire (found it,
    didn't finish the job).
    """
    target, req_verb, req_obj = alf_requirement(task)
    if target is None:
        return "other"
    parsed = [parse_agent_action(a) for a in actions]
    took = any(v == "take" and o == target for v, o in parsed)
    if not took:
        return "find"
    if req_verb is None:
        return "other"  # picked it up and nothing further was required
    if req_verb in _TOGGLE_VERBS:
        req_ok = any((v in _TOGGLE_VERBS) and o == req_obj for v, o in parsed)
    else:
        req_ok = any(v == req_verb and o == req_obj for v, o in parsed)
    return "other" if req_ok else "acquire"


def load_alf_task(gamefile: Path | str) -> dict:
    """Read the ``traj_data.json`` sitting next to a ``game.tw-pddl`` gamefile.

    ``GET /gamefiles`` returns ``*.tw-pddl`` paths; the task JSON (which carries
    ``plan.high_pddl`` + ``pddl_params``) is a sibling in the same trial dir.
    """
    p = Path(gamefile)
    if p.name == "traj_data.json":
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads((p.parent / "traj_data.json").read_text(encoding="utf-8"))


def gamefile_order(env_url: str, seed: int = 1234) -> list[str]:
    """The seed-ordered gamefile list, straight from ``GET /gamefiles?seed=…``.

    ``game_idx`` indexes this list (``env.seed(seed)`` + ``env.skip(game_idx)``
    in the env server), so it is the authoritative ``game_idx -> traj_data.json``
    mapping used to recover each trajectory's expert plan.
    """
    r = httpx.get(f"{env_url}/gamefiles", params={"seed": seed}, timeout=60.0)
    r.raise_for_status()
    return r.json()["gamefiles"]


# ── gridgames: label by replaying the recorded actions ────────────────────────

def gridgames_label(game: str, level: dict, actions: list[str | None]) -> str:
    """Replay a failed gridgame episode -> the authoritative 4-way mode.

    ``actions`` = per-step action strings (``None`` = parse_fail).  A won episode
    returns ``"won"`` (callers only label failures).  ``level`` is the bundle's
    level dict for that ``level_id``.
    """
    env = ta_env.LevelEnv(level)
    for a in actions:
        if env.done:  # the rollout loop stops at done (success / fell_in_hole)
            break
        env.step(a)
    rep = env.report()
    if rep["success"]:
        return "won"
    if game == "frozenlake":
        # "fell_in_hole" is strictly the player actually falling (player -> None);
        # a player who is merely *unreachable* (fl_steps_to_go is None) budgeted
        # out alive and falls through to loop/budget below.
        if env.player is None:
            return "fell_in_hole"
    else:  # sokoban: "dead" == box pushed into an unsolvable position
        if rep["dead"]:
            return "dead_push"
    if has_loop([a for a in actions if a is not None]):
        return "loop"
    return "budget"
