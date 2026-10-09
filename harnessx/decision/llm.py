# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""meta backend: the meta-LLM itself answering the three-axis questions.

This is the *ablation* backend behind ``--decision-backend meta``: instead of a
frozen System-1 read head (kev) scoring each failed trajectory, the *same* strong
meta model (deepseek-v4-pro / local preset) answers the exact same questions
per-trajectory — one generative call that emits a probability vector — and the
runner aggregates those vectors into the identical ``decision_priors.md`` the kev
arm produces.  The only variable changed versus the ``prior`` arm is *who* does
the per-trajectory judgment.

It implements the shared :class:`DiscreteDecisionModel` contract, so
``analyze_trajectories`` / ``build_decision_priors`` / the enforce retrocheck
treat it as a drop-in for kev.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Sequence

from ..core.events import Message
from .types import ChooseQ, Decision, DiscreteDecisionModel, JudgeQ, Question

_log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a failure-mode annotator for agent trajectories. You are given a "
    "compact summary of ONE failed trajectory and a list of questions about it. "
    "Answer every question as a probability (a number between 0 and 1). For a "
    "yes/no question, output p(true). For a multiple-choice question, output an "
    "object mapping each option name to its probability (the values must sum to "
    "1). Respond with ONLY a single JSON object and nothing else (no prose, no "
    "reasoning, no markdown fences). Use keys q0, q1, ... position-aligned with "
    "the questions."
)


class LlmDecisionBackend:
    """Generative LLM (the meta model) as a per-trajectory decision backend.

    ``provider`` is an already-constructed :class:`BaseModelProvider`
    (``AnthropicProvider`` or ``LiteLLMProvider``) — the same instance the runner
    hands to ``MetaAgent``.  Unlike kev/semif (one logit readout per state), each
    ``decide`` is one *generative* call, so this is far more expensive per
    trajectory — that cost is precisely the point of the ablation.
    """

    name = "meta"

    def __init__(self, provider: Any = None, base_url: str | None = None, **kwargs: Any) -> None:
        # ``base_url``/``**kwargs`` are accepted-and-ignored so get_decision_model
        # can instantiate this the same way it instantiates kev/semif.
        if provider is None:
            raise ValueError(
                "LlmDecisionBackend requires `provider` (a BaseModelProvider instance) "
                "so it can call the meta model's `complete`"
            )
        self.provider = provider

    async def __aenter__(self) -> "LlmDecisionBackend":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def decide(self, *, state: str, questions: Sequence[Question]) -> list[Decision]:
        user = self._render_questions(state, questions)
        event = await self.provider.complete(
            messages=[
                Message(role="system", content=_SYSTEM),
                Message(role="user", content=user),
            ],
            tools=[],
        )
        parsed = _parse_json(event.content)
        if parsed is None:
            _log.warning("meta backend: unparseable answer, using neutral priors: %r", (event.content or "")[:200])
        return [_to_decision(i, q, parsed) for i, q in enumerate(questions)]

    @staticmethod
    def _render_questions(state: str, questions: Sequence[Question]) -> str:
        parts = [f"Trajectory summary:\n{state}\n", "Questions:"]
        for i, q in enumerate(questions):
            parts.append(f"\nq{i} — {q.instructions}")
            if isinstance(q, JudgeQ):
                parts.append(
                    f"  (yes/no question; output p(true)): yes = {q.yes_desc or 'yes'}; no = {q.no_desc or 'no'}"
                )
            else:
                opts = "; ".join(f"{name} = {desc}" for name, desc in q.options)
                parts.append(f"  (multiple choice; output a probability distribution over: {opts})")
        return "\n".join(parts)


def _parse_json(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
        return None


def _to_decision(i: int, q: Question, parsed: dict | None) -> Decision:
    raw = parsed.get(f"q{i}") if parsed else None
    if isinstance(q, JudgeQ):
        return Decision.for_judge(_coerce_judge(raw))
    names = [name for name, _ in q.options]
    return Decision.for_choose(_coerce_choose(raw, names))


def _coerce_judge(raw: Any) -> float:
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw)))
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("yes", "true", "y"):
            return 1.0
        if s in ("no", "false", "n"):
            return 0.0
    return 0.5


def _coerce_choose(raw: Any, names: Sequence[str]) -> dict[str, float]:
    if isinstance(raw, dict):
        dist = {name: max(0.0, float(raw.get(name, 0.0))) for name in names}
    elif isinstance(raw, str) and raw.strip() in names:
        dist = {name: 1.0 if name == raw.strip() else 0.0 for name in names}
    else:
        dist = {name: 1.0 / len(names) for name in names}
    total = sum(dist.values())
    if total <= 0:
        dist = {name: 1.0 / len(names) for name in names}
    else:
        dist = {name: v / total for name, v in dist.items()}
    return dist
