# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""System-1 decision-model pre-analysis of rollout trajectories (Mode A).

Before the meta-LLM reads trajectories (SOUL step 3), a cheap discrete-decision
model (kev / semif / anyjev …) reads each *failed* trajectory and emits a
probability vector over benchmark-specific failure modes.  That vector is written
as a machine-rendered prior that the meta-LLM reads *and may override* — it never
replaces the LLM's own diagnosis, it seeds it.

The generic loop lives here.  The benchmark-specific parts — how to parse a
trajectory into compact fields, how to render those fields into the compact text
the model reads, and which failure-mode questions to ask — are supplied by the
caller (see ``recipe/agent_evolver/decision_prior.py``).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..decision import ChooseQ, Decision, JudgeQ, Question, get_decision_model
from ..textutil import clip_text

_log = logging.getLogger(__name__)

ParseFn = Callable[[Path], dict[str, Any] | None]
StateFn = Callable[[dict[str, Any]], str]


@dataclass
class PriorRow:
    """One trajectory's prior: id/labels plus one :class:`Decision` per question."""

    traj_id: str
    goal: str
    eval_reason: str
    decisions: list[Decision]


def _sort_key(path: Path) -> tuple[int, object]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


async def analyze_trajectories(
    *,
    trajectories_dir: Path,
    questions: Sequence[Question],
    parse_trajectory: ParseFn,
    render_state: StateFn,
    model: str = "kev",
    model_kwargs: dict[str, Any] | None = None,
    max_trajectories: int = 96,
    concurrency: int = 1,
) -> list[PriorRow]:
    """Batch-analyze trajectory files through a discrete-decision model.

    ``parse_trajectory`` returns a field dict for failed trajectories (or
    ``None`` to skip — the parser knows its benchmark's success field).
    ``render_state`` turns that dict into the compact text the model reads.
    One ``decide(state, questions)`` call per trajectory (a single forward
    pass, so this is far cheaper than having the meta-LLM re-read each body).

    ``concurrency`` bounds how many ``decide`` calls run in parallel.  kev/semif
    stay at 1 (a logit readout is near-instant); the generative ``meta`` backend
    is ~20s/call, so raising it there collapses the dominant cost.  Result order
    stays deterministic (sorted files, indexed slots).
    """
    files = sorted(trajectories_dir.glob("*.md"), key=_sort_key)[:max_trajectories]
    backend = get_decision_model(model, **(model_kwargs or {}))
    rows: list[PriorRow | None] = [None] * len(files)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(i: int, path: Path) -> None:
        fields = parse_trajectory(path)
        if fields is None:
            return
        async with sem:
            decisions = await backend.decide(state=render_state(fields), questions=questions)
        rows[i] = PriorRow(
            traj_id=path.stem,
            goal=str(fields.get("goal", "")),
            eval_reason=str(fields.get("eval_reason", "")),
            decisions=decisions,
        )

    async with backend:
        await asyncio.gather(*(_one(i, p) for i, p in enumerate(files)))
    return [r for r in rows if r is not None]


def _judge_p(dec: Decision) -> float:
    """p(true) for a judge question (0.0 if absent — shouldn't happen)."""
    return dec.probabilities.get("true", 0.0)


def _choose_cell(dec: Decision) -> str:
    return f"{dec.argmax} ({dec.confidence:.2f})"


def _choose_aggregate(rows: Sequence[PriorRow], idx: int) -> Decision:
    """Round-level mix for a choice question: mean probability per option → argmax."""
    totals: dict[str, float] = {}
    for r in rows:
        for name, p in r.decisions[idx].probabilities.items():
            totals[name] = totals.get(name, 0.0) + p
    n = len(rows) or 1
    return Decision.for_choose({name: tot / n for name, tot in totals.items()})


def aggregate_choices(*, rows: Sequence[PriorRow], idx: int) -> Decision:
    """Round-level argmax of a ChooseQ column (for ``enforce`` runners).

    Raises ``ValueError`` on empty ``rows`` — no trajectories means no prior.
    """
    if not rows:
        raise ValueError("cannot aggregate a choice over zero trajectories")
    return _choose_aggregate(rows, idx)


def render_priors_markdown(
    *,
    rows: Sequence[PriorRow],
    questions: Sequence[Question],
    names: Sequence[str],
) -> str:
    """Render the Markdown priors file the meta-agent reads in SOUL step 3.

    ``questions`` and ``names`` are position-aligned.  A :class:`JudgeQ` column
    renders p(true); a :class:`ChooseQ` column renders ``argmax (confidence)``.
    """
    n = len(names)
    is_choice = [isinstance(q, ChooseQ) for q in questions]
    head = [
        "| id | goal | eval | " + " | ".join(names) + " |",
        "|----|------|------|" + "---|" * n,
    ]
    body: list[str] = []
    if rows:
        agg_cells: list[str] = []
        for i, choice in enumerate(is_choice):
            if choice:
                agg_cells.append(_choose_cell(_choose_aggregate(rows, i)))
            else:
                p = sum(_judge_p(r.decisions[i]) for r in rows) / len(rows)
                agg_cells.append(f"{p:.2f}")
        body.append("| **aggregate** | — | — | " + " | ".join(agg_cells) + " |")
    for r in rows:
        cells = [r.traj_id, clip_text(r.goal, 30), r.eval_reason]
        for i, choice in enumerate(is_choice):
            cells.append(_choose_cell(r.decisions[i]) if choice else f"{_judge_p(r.decisions[i]):.2f}")
        body.append("| " + " | ".join(cells) + " |")

    return (
        "# System-1 trajectory priors\n\n"
        f"{len(rows)} failed trajectory(s) pre-analyzed by a cheap System-1 "
        "decision model. Judge columns are p(true); choice columns are "
        "`argmax (confidence)`; the aggregate row is the round-level mix.\n\n"
        "**Priors, not ground truth.** Use them to seed step-3 diagnosis, then "
        "verify against the raw trajectories and override where you disagree.\n\n"
        + "\n".join(head + body)
        + "\n"
    )
