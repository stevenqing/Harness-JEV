# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Failure types for the layered-candidates loop (§4 of ``specs/kev_ranker_step0c.md``).

Two jobs, one kev pass per round:

1. **Diagnostic** (§4.1 / §4.2) — for every failed *parent* trajectory, record
   the ground-truth label *and* kev's answer, then the per-round accuracy.  This
   is the A1/A2 accuracy line, continued through the evolution.

   - ALFWorld ground truth = ``find`` / ``acquire`` / ``other``, recovered from
     ``traj_data.json`` (``plan.high_pddl``) exactly as ``a1a2_runner.py`` does.
     kev answers the two A1/A2 JudgeQs (``reviewer.ALF_QUESTIONS``); the tag is
     find/acquire, else ``other`` (neither clears 0.5).
   - WebShop ground truth = the six programmatic stage labels
     (``webshop_labels.ws_label``).  kev answers one six-way ``ChooseQ``.

2. **Brief structure** (§4.4) — the machine-rendered prior the meta-agent reads.
   Fixed question set: the failure-type question(s) plus the 4-layer ``LEVER_Q``
   (configuration / control / action / instruction, recovered from the retired
   three-axis ``AXIS_QUESTIONS``).  The three arms differ only in the *values*
   (llm = none, const = 0.5 everywhere, kev = kev's answers), not the structure.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from harnessx.decision import ChooseQ, Decision
from harnessx.meta_harness.decision_prior import (
    analyze_trajectories,
    render_priors_markdown,
)

from . import reviewer
from .decision_prior import _parse_trajectory, _render_state, _sort_key
from .failure_labels import alfworld_label, gamefile_order, load_alf_task
from .webshop_labels import ws_labels_from_records

_log = logging.getLogger(__name__)

# ── 4-layer lever ChooseQ (recovered from the retired three-axis AXIS_QUESTIONS) ──
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

WS_QUESTION = ChooseQ(
    "Which failure stage did this WebShop trajectory reach?",
    options=(
        ("no_search", "never issued a search[] action at all"),
        ("not_bought", "searched but never clicked buy within the step budget"),
        ("wrong_type", "bought an item whose type does not match the goal"),
        ("attr_mismatch", "bought the right type but missed some required attributes"),
        ("option_mismatch", "right type and attributes but missed some required options"),
        ("price_over", "right item but the price exceeded the budget"),
    ),
)


def failure_type_questions(benchmark: str) -> list:
    """The failure-type question(s) only (for the diagnostic tag)."""
    if benchmark == "alfworld":
        return list(reviewer.ALF_QUESTIONS)
    if benchmark == "webshop":
        return [WS_QUESTION]
    raise NotImplementedError(f"failure types not wired for {benchmark!r}")


def layered_questions(benchmark: str) -> list:
    """§4.4 brief question set: failure-type question(s) + the 4-layer lever."""
    return [*failure_type_questions(benchmark), LEVER_Q]


def layered_names(benchmark: str) -> list[str]:
    if benchmark == "alfworld":
        return ["find", "acquire", "lever"]
    if benchmark == "webshop":
        return ["failure_stage", "lever"]
    raise NotImplementedError(f"failure types not wired for {benchmark!r}")


# ── ground truth ──────────────────────────────────────────────────────────────

def alf_ground_truth(env_url: str, seed: int, trajectories_dir: Path) -> dict[str, str]:
    """``find``/``acquire``/``other`` for every failed ALFWorld trajectory.

    Same recovery as ``a1a2_runner.py``: ``traj_id`` (= game_idx) → the seed-
    ordered gamefile list → ``traj_data.json`` → ``alfworld_label``.
    """
    order = gamefile_order(env_url, seed)
    labels: dict[str, str] = {}
    files = sorted(trajectories_dir.glob("*.md"), key=_sort_key)
    for p in files:
        fields = _parse_trajectory(p)
        if fields is None:
            continue
        try:
            task = load_alf_task(order[int(p.stem)])
        except (ValueError, IndexError, KeyError):
            _log.warning("cannot resolve traj_data.json for game_idx %s", p.stem)
            continue
        labels[p.stem] = alfworld_label([a for a, _ in fields["pairs"]], task)
    return labels


def ws_ground_truth(records: list[dict]) -> dict[str, str]:
    """``{task_id: label}`` from one rollout's in-memory records (needs ``reward_info``)."""
    return {str(tid): label for tid, label in ws_labels_from_records(records).items()}


# ── tag extraction ────────────────────────────────────────────────────────────

def kev_tag(benchmark: str, decisions: list) -> str:
    """Failure-type tag from kev's answers (position-aligned to ``failure_type_questions``)."""
    if benchmark == "alfworld":
        scores = {
            "find": decisions[0].probabilities.get("true", 0.0),
            "acquire": decisions[1].probabilities.get("true", 0.0),
        }
        return reviewer.tag_from_scores(scores, ("find", "acquire"))
    if benchmark == "webshop":
        return decisions[0].argmax or "other"
    raise NotImplementedError(benchmark)


# ── one-pass analysis (diagnostic + brief share the same kev forward passes) ──

async def analyze_parent_failures(
    *,
    benchmark: str,
    trajectories_dir: Path,
    env_url: str | None = None,
    seed: int = 1234,
    model: str = "kev",
    model_kwargs: dict | None = None,
    ws_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run kev over the failure-type + lever questions; recover ground truth.

    Returns::

        {
          "questions": list, "names": list[str], "rows": list[PriorRow],
          "labels": dict[traj_id, str], "kev_tags": dict[traj_id, str],
          "accuracy": float, "label_distribution": dict[str, int],
          "kev_distribution": dict[str, int],
        }
    """
    questions = layered_questions(benchmark)
    names = layered_names(benchmark)

    # ground truth.  WebShop labels need the in-memory ``reward_info`` (the
    # record's trajectory/reward_info are popped after ``write_task_trajectory``),
    # so the caller pre-computes them from the live records and passes ``ws_labels``.
    if benchmark == "alfworld":
        labels = alf_ground_truth(env_url, seed, trajectories_dir)
    elif benchmark == "webshop":
        labels = {str(k): v for k, v in (ws_labels or {}).items()}
    else:
        raise NotImplementedError(benchmark)

    rows, _judging = await analyze_trajectories(
        trajectories_dir=trajectories_dir,
        questions=questions,
        parse_trajectory=_parse_trajectory,
        render_state=_render_state,
        model=model,
        model_kwargs=model_kwargs,
        max_trajectories=100_000,
    )

    kev_tags: dict[str, str] = {}
    for r in rows:
        kev_tags[r.traj_id] = kev_tag(benchmark, r.decisions)

    common = [tid for tid in labels if tid in kev_tags]
    truth = [labels[t] for t in common]
    pred = [kev_tags[t] for t in common]
    accuracy = reviewer.accuracy(truth, pred)

    return {
        "questions": questions,
        "names": names,
        "rows": rows,
        "labels": labels,
        "kev_tags": kev_tags,
        "accuracy": accuracy,
        "label_distribution": dict(Counter(labels.values())),
        "kev_distribution": dict(Counter(kev_tags.values())),
    }


def render_brief_markdown(
    analysis: dict[str, Any],
    *,
    constant_prior: float | None = None,
    judging_seconds: float | None = None,
) -> str:
    """Render the §4.4 brief markdown from an ``analyze_parent_failures`` result.

    ``constant_prior`` (the ``const`` arm) neutralizes every decision after the
    scoring pass — judge columns become p(true)=``constant_prior``, choice
    columns become uniform over their options — erasing values while keeping the
    question structure.
    """
    rows = analysis["rows"]
    if constant_prior is not None:
        c = float(constant_prior)
        questions = analysis["questions"]
        for row in rows:
            for i, dec in enumerate(row.decisions):
                q = questions[i] if i < len(questions) else None
                if isinstance(q, ChooseQ):
                    # neutral choice = uniform over its option names (structure kept, signal erased)
                    names = [name for name, _desc in q.options]
                    row.decisions[i] = Decision.for_choose({n: 1.0 / len(names) for n in names})
                else:
                    # neutral judge = p(true) = c (0.5 for the const arm)
                    row.decisions[i] = Decision.for_judge(c)
    return render_priors_markdown(
        rows=rows,
        questions=analysis["questions"],
        names=analysis["names"],
        judging_seconds=judging_seconds,
    )


def write_failure_types(analysis: dict[str, Any], out_path: Path) -> None:
    """Persist the per-round diagnostic table (labels + kev answers) as JSON.

    In addition to the coarse tag, the full per-question decision distribution
    is stored (``kev_decisions``, position-aligned to ``names``) so §7's exporter
    can recover ``kev_answer`` without re-scoring or parsing the markdown brief.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_id = {r.traj_id: r for r in analysis["rows"]}

    def _decision(d: Any) -> dict:
        return {
            "argmax": d.argmax,
            "confidence": d.confidence,
            "probabilities": d.probabilities,
        }

    per_traj = []
    for tid in sorted(analysis["labels"]):
        row = rows_by_id.get(tid)
        per_traj.append({
            "traj_id": tid,
            "label": analysis["labels"].get(tid),
            "kev_tag": analysis["kev_tags"].get(tid),
            "kev_decisions": [_decision(d) for d in row.decisions] if row else None,
        })
    payload = {
        "names": analysis["names"],
        "accuracy": analysis["accuracy"],
        "label_distribution": analysis["label_distribution"],
        "kev_distribution": analysis["kev_distribution"],
        "per_trajectory": per_traj,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
