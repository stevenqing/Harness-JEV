# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""System-1 reviewer vocabulary + agreement metrics (visibility A1/A2).

kev is asked one independent noul (判断) per failure mode — the noul *vector*,
not a single forced choice (the bucketing history showed noul >> 4-way choice,
and the head is a noul vector).  The tag is the argmax over the modes; each
mode's p(true) is kept as a score so A1 can report the AUC of the ``find`` score
against the ``find`` label.

The metrics are pure and dependency-free (no sklearn) so the report is
recomputable anywhere.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from harnessx.decision import JudgeQ

# ── reviewer vocabulary (closed-set failure modes per benchmark) ─────────────

ALF_MODES = ("find", "acquire")
ALF_QUESTIONS = [
    JudgeQ(
        "Did the agent fail to find/acquire the target object at all — it never "
        "issued `take <target>` and thrashed searching instead?",
        yes_desc="yes: never acquired the target (find failure)",
        no_desc="no: the agent did acquire the target",
    ),
    JudgeQ(
        "Did the agent acquire the target object but fail to complete the next "
        "required action (clean/heat/cool/put/slice/use/toggle)?",
        yes_desc="yes: acquired but didn't finish the next action (acquire failure)",
        no_desc="no: not an acquire failure",
    ),
]

GRIDGAMES_MODES = ("fell_in_hole", "dead_push", "loop", "budget")
GRIDGAMES_QUESTIONS = [
    JudgeQ("Did the player fall into a hole (FrozenLake)?", "yes: fell in a hole", "no"),
    JudgeQ("Did the agent push a box into a dead wall/corner position (Sokoban)?", "yes: dead push", "no"),
    JudgeQ("Did the agent loop on a repeated direction without making progress?", "yes: looped", "no"),
    JudgeQ("Did the agent run out of moves without dying or looping?", "yes: budget", "no"),
]


def tag_from_scores(scores: dict[str, float], modes: tuple[str, ...], threshold: float = 0.5) -> str:
    """Argmax tag over ``modes``; ``"other"`` when no mode clears ``threshold``."""
    top = max(modes, key=lambda m: scores.get(m, 0.0))
    return top if scores.get(top, 0.0) >= threshold else "other"


# ── agreement metrics ─────────────────────────────────────────────────────────

def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    n = len(y_true)
    return sum(t == p for t, p in zip(y_true, y_pred)) / n if n else float("nan")


def cohens_kappa(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Cohen's kappa (observed agreement corrected for chance)."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    po = sum(t == p for t, p in zip(y_true, y_pred)) / n
    ct, cp = Counter(y_true), Counter(y_pred)
    pe = sum(ct[k] * cp[k] for k in set(ct) | set(cp)) / (n * n)
    return (po - pe) / (1.0 - pe) if pe < 1.0 else 0.0


def auc(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    """ROC AUC by the Mann-Whitney U statistic (ties count half).  NaN when one
    class is absent."""
    pos = [s for s, t in zip(scores, y_true) if t]
    neg = [s for s, t in zip(scores, y_true) if not t]
    if not pos or not neg:
        return float("nan")
    total = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                total += 1.0
            elif p == q:
                total += 0.5
    return total / (len(pos) * len(neg))
