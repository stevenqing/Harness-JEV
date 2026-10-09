# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Deterministic rule-tagger trajectory priors (visibility spec A3 ``rule`` arm).

The A3 three-arm comparison injects three different System-1 analyses into the
meta-evolve brief: ``llm`` (none — the meta-LLM diagnoses itself), ``kev`` (the
frozen pointer-head model's three-axis readout), and ``rule`` (a *deterministic*
text-only tagger — the A2 baseline).  This module is the ``rule`` arm: it reads
the same failed trajectories and writes the same *kind* of Markdown prior, but
with a hard failure-mode tag per trajectory instead of a probability vector.

The tag is exactly the A2 ``rule_tagger`` output (``alfworld_rule`` for ALFWorld,
``gridgames_rule`` for gridgames), so the ``rule`` arm is held to the same
information ceiling as kev — and its blindness to *which* object the goal wants
is the place kev's semantic read is predicted to win (prediction P3).

The file shape mirrors ``render_priors_markdown`` (id / goal / eval columns plus
one mode column and an aggregate distribution row), so the meta-agent reads it
through the same ``decision_priors_path`` channel as the kev prior.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .decision_prior import _parse_trajectory
from .rule_tagger import alfworld_rule, gridgames_rule

_RULE_MODES = {
    "alfworld": ("find", "acquire", "other"),
    "gridgames": ("fell_in_hole", "dead_push", "loop", "budget"),
}


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _alf_rule(fields: dict) -> str:
    return alfworld_rule([a for a, _ in fields["pairs"]])


def _gg_rule(fields: dict, game: str) -> str:
    acts = [None if inv else a for a, inv in fields["pairs"]]
    return gridgames_rule(game, fields["eval_reason"], acts)


def build_rule_priors(
    *,
    benchmark: str,
    trajectories_dir: Path,
    output_path: Path,
    game: str | None = None,
) -> Path:
    """Tag each failed trajectory with the deterministic A2 rule; write the prior.

    ``game`` is required for gridgames (to pick the fell_in_hole vs dead_push
    semantics); ALFWorld does not need it.
    """
    if benchmark not in _RULE_MODES:
        raise NotImplementedError(
            f"rule prior not wired for benchmark {benchmark!r}; available: {sorted(_RULE_MODES)}"
        )
    modes = _RULE_MODES[benchmark]

    files = sorted(trajectories_dir.glob("*.md"), key=lambda p: (int(p.stem) if p.stem.isdigit() else 1e9, p.stem))
    rows: list[tuple[str, str, str, str]] = []  # (id, goal, eval, mode)
    for p in files:
        fields = _parse_trajectory(p)
        if fields is None:
            continue
        if benchmark == "alfworld":
            mode = _alf_rule(fields)
        else:
            mode = _gg_rule(fields, game or "frozenlake")
        rows.append((p.stem, _clip(fields["goal"], 30), fields["eval_reason"], mode))

    dist = Counter(r[3] for r in rows)
    agg = ", ".join(f"{m}={dist.get(m, 0)}" for m in modes)
    head = ["| id | goal | eval | rule_mode |", "|----|------|------|-----------|"]
    body: list[str] = []
    if rows:
        body.append(f"| **aggregate** | — | — | {agg} |")
    for traj_id, goal, ev, mode in rows:
        body.append(f"| {traj_id} | {goal} | {ev} | {mode} |")

    text = (
        "# Deterministic rule-tagger trajectory priors\n\n"
        f"{len(rows)} failed trajectory(s) tagged by a deterministic text-only "
        "rule (the A2 baseline). The tag is a hard label, not a probability — "
        "it has no confidence and, for ALFWorld, is deliberately blind to *which* "
        "object the goal asks for.\n\n"
        "**Priors, not ground truth.** Use them to seed step-3 diagnosis, then "
        "verify against the raw trajectories and override where you disagree.\n\n"
        + "\n".join(head + body)
        + "\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path
