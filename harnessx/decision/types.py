# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Canonical decision vocabulary shared by every backend.

The whole package pivots on two closed-set question shapes:

- :class:`JudgeQ` — 判断: a yes/no question, answered as ``p(true)``.
- :class:`ChooseQ` — 选项: argmax over a closed set of named candidates.

:class:`Decision` is the backend-agnostic answer (a probability distribution plus
an argmax and an abstention-style confidence).  Backends convert *into* and *out
of* this vocabulary; callers never see a backend's native wire format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, Union


@dataclass(frozen=True)
class JudgeQ:
    """判断: a closed-set yes/no question.

    ``yes_desc``/``no_desc`` name what the two outcomes *mean* in this context —
    they anchor the model's interpretation (kev renders them as the option
    descriptions).  Empty is a valid "bare yes/no".
    """

    instructions: str = ""
    yes_desc: str = ""
    no_desc: str = ""


@dataclass(frozen=True)
class ChooseQ:
    """选项: argmax over a closed set of named candidates.

    ``options`` is an order-preserving ``(name, description)`` sequence; the
    returned :class:`Decision` reports probabilities keyed by ``name``.
    """

    instructions: str = ""
    options: tuple[tuple[str, str], ...] = ()


Question = Union[JudgeQ, ChooseQ]


@dataclass
class Decision:
    """One question's answer as a probability distribution.

    - ``probabilities`` — ``{"false", "true"}`` for a judge; ``{name: p}`` for a choice.
    - ``argmax`` — the modal label (``None`` never occurs in practice; every
      question is closed-set with ≥1 candidate).
    - ``confidence`` — a 0..1 abstention signal: 1.0 = unambiguous, 0.0 = a
      coin-flip.  Callers can treat a low value as "fall back to the LLM".
    """

    probabilities: dict[str, float]
    argmax: str | None = None
    confidence: float = 0.0

    @classmethod
    def for_judge(cls, p_true: float) -> "Decision":
        """Build a judge answer from ``p(true)``; confidence = distance from a coin-flip."""
        p_true = float(p_true)
        return cls(
            probabilities={"false": 1.0 - p_true, "true": p_true},
            argmax="true" if p_true >= 0.5 else "false",
            confidence=abs(p_true - 0.5) * 2.0,
        )

    @classmethod
    def for_choose(cls, dist: dict[str, float]) -> "Decision":
        """Build a choice answer; confidence = how far the mode clears a uniform prior.

        Mirrors kev's ``choice_confidence``: ``(max(p) - 1/K) / (1 - 1/K)``.
        """
        dist = {k: float(v) for k, v in dist.items()}
        k = len(dist)
        if k == 0:
            raise ValueError("a choice needs at least one candidate")
        top = max(dist, key=dist.get)
        conf = 1.0 if k == 1 else (max(dist.values()) - 1.0 / k) / (1.0 - 1.0 / k)
        return cls(probabilities=dist, argmax=top, confidence=max(0.0, min(1.0, conf)))


class DiscreteDecisionModel(Protocol):
    """A System-1 discrete-decision model: one ``state`` → N closed-set questions in one pass.

    Backends are free to implement this over any transport (an HTTP sidecar, an
    in-process logit readout, …).  The contract is deliberately the *batch* form
    (``decide(state, [q1, q2, …])``): every System-1 model reads one state against
    many candidates in a single forward pass, so per-question calls would waste a
    forward pass each.
    """

    name: str

    async def decide(self, *, state: str, questions: Sequence[Question]) -> list[Decision]:
        """Answer ``questions`` about ``state``; returns a :class:`Decision` per question, position-aligned."""
        ...
