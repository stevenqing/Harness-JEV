# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Benchmark-agnostic System-1 trajectory priors for the evolver.

The failure-mode vocabulary is now the *harness framework's own axes* (from
``analyze/SKILL.md``) — lens (看什么) × lever (改什么) — rather than a
per-benchmark list of failure modes (the old ``find/acquire/malformed``).  Those
axes are benchmark-agnostic, so the questions are defined once here.

What stays per-benchmark is the *trajectory parser/renderer*: how a ``.md``
trajectory is turned into the compact state text the model reads.  ALFWorld and
gridgames share the same ``write_task_trajectory`` shape (``**Extracted
action**`` body); WebShop has its own and is not wired here yet.

``build_decision_priors`` is the single entry point the runners call right
before ``MetaAgent.evolve()``.  It returns a :class:`DecisionPriors` (the
written file path plus the aggregate lens/lever readout) so an ``enforce``
runner can read the binding lever decision directly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from harnessx.decision import ChooseQ, JudgeQ
from harnessx.meta_harness.decision_prior import (
    aggregate_choices,
    analyze_trajectories,
    render_priors_markdown,
)

_log = logging.getLogger(__name__)


# ── benchmark-agnostic three-axis vocabulary (lens × lever) ─────────────────
# lens — 看什么: what kind of pattern is this failed trajectory
LENS_FAILURE_Q = JudgeQ(
    "Did the agent fail because a blocker kept it from finishing — it ran out of "
    "steps/budget, looped on repeated actions, hit an error, or committed a wrong "
    "final answer (a mechanical / commitment failure)?",
    yes_desc="yes: mechanical/commitment failure (blocked from finishing)",
    no_desc="no: not a mechanical failure",
)
LENS_CAPABILITY_GAP_Q = JudgeQ(
    "Did the agent try to do something it lacked the capability/tool/knowledge to do "
    "(it could not perform a class of action at all, rather than making a reasoning "
    "or mechanics error)?",
    yes_desc="yes: capability gap (tried but couldn't)",
    no_desc="no: not a capability gap",
)
# lever — 改什么: argmax over the 4 framework levers
LEVER_Q = ChooseQ(
    "Where does the fix for this trajectory's failure belong? Pick the single most "
    "appropriate lever.",
    options=(
        ("configuration", "tune an existing knob/parameter (budget, window, threshold)"),
        ("control", "add/change a mechanical processor hook (parse/sanitize/reformat/guard)"),
        ("action", "add a new tool — the agent lacks a way to take a class of action"),
        ("instruction", "change the system prompt / guidance / skill — the agent doesn't know when/how to act"),
    ),
)

AXIS_QUESTIONS = [LENS_FAILURE_Q, LENS_CAPABILITY_GAP_Q, LEVER_Q]
AXIS_NAMES = ["lens_failure", "lens_capability_gap", "lever"]

# column index of the lever ChooseQ (read the binding lever decision)
LEVER_IDX = AXIS_QUESTIONS.index(LEVER_Q)


# ── shared trajectory parser/renderer (ALF/gridgames write_task_trajectory) ──
_FIELD_RE = re.compile(r"^(\w+):\s*(.*)$", re.M)
_ACTION_RE = re.compile(r"\*\*Extracted action\*\*:\s*`(.*?)`\s*\(invalid=(True|False)\)")


def _parse_trajectory(path: Path) -> dict[str, Any] | None:
    """→ {goal, exit_reason, eval_reason, steps, pairs, final_obs} for failed episodes; None for won/skip."""
    text = path.read_text(encoding="utf-8")
    fm = dict(_FIELD_RE.findall(text[: text.find("\n---\n")]))
    eval_reason = fm.get("eval_reason", "").strip('"')
    if eval_reason == "won":
        return None
    pairs = [(m.group(1), m.group(2) == "True") for m in _ACTION_RE.finditer(text)]
    return {
        "goal": fm.get("goal", "").strip('"'),
        "exit_reason": fm.get("exit_reason", "").strip('"'),
        "eval_reason": eval_reason,
        "steps": fm.get("steps", "").strip('"'),
        "final_obs": fm.get("final_obs", "").strip('"'),
        "pairs": pairs,
    }


def _render_state(t: dict[str, Any]) -> str:
    acts = "; ".join(f"{a} [invalid]" if inv else a for a, inv in t["pairs"])
    if len(acts) > 900:
        acts = acts[:900] + "…"
    return (
        f"goal: {t['goal']}\n"
        f"exit_reason: {t['exit_reason']}  eval_reason: {t['eval_reason']}  steps: {t['steps']}\n"
        f"actions: {acts}\n"
    )


# benchmark → (parse, render); questions are universal now
_PARSERS: dict[str, tuple[Callable[[Path], dict[str, Any] | None], Callable[[dict[str, Any]], str]]] = {
    "alfworld": (_parse_trajectory, _render_state),
    "gridgames": (_parse_trajectory, _render_state),
    # WebShop trajectories use the same write_task_trajectory shape (same
    # frontmatter fields + "**Extracted action**" body), so the shared parser
    # applies unchanged; only the action vocabulary differs (search[]/click[]).
    "webshop": (_parse_trajectory, _render_state),
}


@dataclass
class DecisionPriors:
    """Result of a prior build: the written file plus the aggregate readout."""

    path: Path
    lever_argmax: str | None
    lever_confidence: float
    lens_failure_p: float
    lens_capability_gap_p: float


async def build_decision_priors(
    *,
    benchmark: str,
    trajectories_dir: Path,
    output_path: Path,
    model: str = "kev",
    base_url: str = "http://127.0.0.1:8090",
    max_trajectories: int = 96,
    model_kwargs: dict | None = None,
    concurrency: int = 1,
) -> DecisionPriors:
    """Render benchmark-agnostic three-axis priors for a benchmark's trajectories.

    ``model`` is any registered backend name (default ``"kev"``); pass the same
    name to the runner's ``--decision-backend`` so they stay in sync.  For the
    ``"meta"`` backend, pass ``model_kwargs={"provider": meta_provider}`` so the
    backend can call the meta model itself, and ``concurrency`` > 1 to run the
    per-trajectory generative judgments in parallel.
    """
    try:
        parse_trajectory, render_state = _PARSERS[benchmark]
    except KeyError:
        raise NotImplementedError(
            f"decision priors not wired for benchmark {benchmark!r}; available: {sorted(_PARSERS)}"
        ) from None

    kwargs = dict(model_kwargs or {})
    kwargs.setdefault("base_url", base_url)
    rows = await analyze_trajectories(
        trajectories_dir=trajectories_dir,
        questions=AXIS_QUESTIONS,
        parse_trajectory=parse_trajectory,
        render_state=render_state,
        model=model,
        model_kwargs=kwargs,
        max_trajectories=max_trajectories,
        concurrency=concurrency,
    )
    if not rows:
        _log.warning("no failed trajectories to analyze under %s; priors will be empty", trajectories_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_priors_markdown(rows=rows, questions=AXIS_QUESTIONS, names=AXIS_NAMES),
        encoding="utf-8",
    )

    if rows:
        lever = aggregate_choices(rows=rows, idx=LEVER_IDX)
        lens_failure = sum(r.decisions[0].probabilities["true"] for r in rows) / len(rows)
        lens_capability = sum(r.decisions[1].probabilities["true"] for r in rows) / len(rows)
        return DecisionPriors(
            path=output_path,
            lever_argmax=lever.argmax,
            lever_confidence=lever.confidence,
            lens_failure_p=lens_failure,
            lens_capability_gap_p=lens_capability,
        )
    return DecisionPriors(
        path=output_path,
        lever_argmax=None,
        lever_confidence=0.0,
        lens_failure_p=0.0,
        lens_capability_gap_p=0.0,
    )
