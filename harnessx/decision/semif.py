# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""semif backend: frozen Qwen3.5-4B first-token logit readout.

SemIf-style direct scoring (github.com/TheoLeeCJ/SemIf): options are mapped to
single uppercase letters and the model's next-token logits at those letter slots
are read directly, softmaxed, and argmaxed — one forward pass per decision, no
decoding loop, no JSON repair.  The transport is vLLM's OpenAI-compatible
``logprobs`` on the same frozen base weights.

This is the second ``jevlike`` backend behind the shared
:class:`DiscreteDecisionModel` interface.  It answers :class:`ChooseQ` natively
(one letter per option) and :class:`JudgeQ` as a two-option (no/yes → A/B)
readout.

Unlike :class:`KevBackend` (one batch pass for N questions), every question here
is its own forward pass — the letter readout is single-criterion by design.
Callers that batch many questions per state should prefer kev.
"""

from __future__ import annotations

import math
from typing import Sequence

import httpx

from .types import ChooseQ, Decision, DiscreteDecisionModel, JudgeQ, Question

# vLLM caps top_logprobs at --max-logprobs (served with 40).  The rarest option
# letters only survive the readout if we request the maximum K; logit_bias alone
# does not change the *returned* top_logprobs in vLLM 0.28+, so both are used.
TOP_LOGPROBS = 40

# Qwen3.5-4B tokenizer ids for A..Z (32..57, consecutive — verified single
# round-trip tokens).  logit_bias forces each declared letter into the top-k
# readout; a common shift preserves relative letter logits, so the softmax is
# unchanged.
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LETTER_IDS = {ch: 32 + i for i, ch in enumerate(LETTERS)}
_MAX_OPTIONS = len(LETTERS) - 6  # 20 options cap (mirrors the SemIf MAX_OPTIONS)

_DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one "
    "listed option. Respond with only its uppercase letter, with no explanation "
    "or reasoning."
)


class SemifBackend:
    """Frozen base-model first-token logit readout over a vLLM ``/v1`` endpoint.

    Usage (async context manager owns the connection pool)::

        async with SemifBackend(base_url="http://127.0.0.1:8200/v1") as semif:
            ds = await semif.decide(
                state="Board: ...\\nMoves left: 8",
                questions=[ChooseQ("which move?", (("up", "move up"), ("down", "move down")))],
            )
    """

    name = "semif"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8200/v1",
        model: str = "Qwen3.5-4B",
        api_key: str = "EMPTY",
        timeout: httpx.Timeout | None = None,
        limits: httpx.Limits | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._timeout = timeout or httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
        self._limits = limits or httpx.Limits(max_connections=256, max_keepalive_connections=64)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SemifBackend":
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout, limits=self._limits)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("SemifBackend is not entered; use `async with SemifBackend(...)`")
        return self._client

    async def decide(self, *, state: str, questions: Sequence[Question]) -> list[Decision]:
        client = self._ensure_client()
        out: list[Decision] = []
        for q in questions:
            if isinstance(q, JudgeQ):
                descs = [q.no_desc or "no", q.yes_desc or "yes"]
                probs = await self._score(client, state, q.instructions, descs)
                out.append(Decision.for_judge(probs[1]))
            else:
                names = [name for name, _ in q.options]
                descs = [desc for _, desc in q.options]
                probs = await self._score(client, state, q.instructions, descs)
                out.append(Decision.for_choose({names[i]: probs[i] for i in range(len(names))}))
        return out

    async def _score(
        self,
        client: httpx.AsyncClient,
        evidence: str,
        criterion: str,
        options: Sequence[str],
    ) -> list[float]:
        """Score ``options`` via direct next-token letter logits → softmax probs.

        Letters missing from the returned top-k read -1e9 (effectively 0 prob).
        """
        options = list(options[:_MAX_OPTIONS])
        payload = {
            "evidence": evidence,
            "criterion": criterion,
            "options": [
                {"letter": LETTERS[i], "description": opt} for i, opt in enumerate(options)
            ],
        }
        msgs = [
            {"role": "system", "content": _DIRECT_SYSTEM},
            {"role": "user", "content": _json_dumps(payload)},
        ]
        logit_bias = {_LETTER_IDS[LETTERS[i]]: 100.0 for i in range(len(options))}
        body = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
            "temperature": 0.0,
            "logit_bias": logit_bias,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = await client.post("/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        # top_logprobs is sorted by descending logprob; first occurrence per
        # stripped key wins so a later " J" cannot overwrite the genuine "J".
        logp: dict[str, float] = {}
        for t in top:
            logp.setdefault(t["token"].strip(), t["logprob"])
        scores = [logp.get(LETTERS[i], -1e9) for i in range(len(options))]
        return _softmax(scores)


def _json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


def _softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    w = [math.exp(s - m) for s in scores]
    total = sum(w)
    return [x / total for x in w]
