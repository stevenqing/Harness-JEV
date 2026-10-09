# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""kev backend: a thin async client over the ``POST /v1/systemone`` sidecar.

kev exposes the general discrete-decision primitive directly (``noul`` ≈ 判断,
``choice`` ≈ 选项), so this adaptor is a near-1:1 mapping of the canonical
vocabulary onto kev's wire format.  It is the reference backend — the first of
several ``jevlike`` models behind the same :class:`DiscreteDecisionModel`
interface.
"""

from __future__ import annotations

import logging
from typing import Sequence

import httpx

from .types import ChooseQ, Decision, DiscreteDecisionModel, JudgeQ, Question

_log = logging.getLogger(__name__)


class KevBackend:
    """Async client for a kev ``/v1/systemone`` sidecar.

    Usage (async context manager owns the connection pool)::

        async with KevBackend(base_url="http://127.0.0.1:8090") as kev:
            ds = await kev.decide(
                state="exit_reason: max_steps\\nmoves: up, up, up",
                questions=[JudgeQ("is the agent collapsing to one direction?", "yes", "no")],
            )
    """

    name = "kev"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8090",
        model: str = "kev-latest",
        timeout: httpx.Timeout | None = None,
        limits: httpx.Limits | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = timeout or httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)
        self._limits = limits or httpx.Limits(max_connections=64, max_keepalive_connections=16)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "KevBackend":
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout, limits=self._limits)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("KevBackend is not entered; use `async with KevBackend(...)`")
        return self._client

    @staticmethod
    def _to_kev_question(q: Question) -> dict:
        if isinstance(q, JudgeQ):
            return {
                "type": "noul",
                "instructions": q.instructions,
                "criteria": {"false": q.no_desc, "true": q.yes_desc},
            }
        return {
            "type": "choice",
            "instructions": q.instructions,
            "criteria": {name: desc for name, desc in q.options},
        }

    async def decide(self, *, state: str, questions: Sequence[Question]) -> list[Decision]:
        client = self._ensure_client()
        body = {
            "state": state,
            "model": self.model,
            "questions": {f"q{i}": self._to_kev_question(q) for i, q in enumerate(questions)},
        }
        resp = await client.post("/v1/systemone", json=body)
        resp.raise_for_status()
        answers = resp.json()["answers"]

        out: list[Decision] = []
        for i, q in enumerate(questions):
            ans = answers[f"q{i}"]
            if isinstance(q, JudgeQ):
                out.append(Decision.for_judge(ans["noul"]))
            else:
                out.append(Decision.for_choose(ans["probabilities"]))
        return out
