# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Candidate-level retroactive-check scoring (intent axis) for enforce mode.

After the meta-LLM drafts candidates in ``candidates.md``, a cheap System-1
decision model (kev) reads each ``## Candidate C-NNN`` and scores its
retroactive check — the *intent* axis of the framework's lens × lever × intent
triad:

- corrective / preservative-transfer → "would the failing trajectory flip to
  pass under this change?" (retroactive-check variant A / C);
- preservative-lock → "would removing this habit regress the passing
  trajectory?" (variant B).

This is the *hard* half of ``--decision-mode enforce``: candidates kev judges
"no" (``pass_prob < reject_threshold``) are rejected and the round reverts.
Unlike the lever gate (which binds one lever for the whole round), the intent
axis is *per-candidate* — each candidate carries its own retroactive question.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from harnessx.decision import JudgeQ, get_decision_model
from harnessx.textutil import clip_text

_log = logging.getLogger(__name__)

_CANDIDATE_RE = re.compile(r"^##\s+Candidate\s+(C-\d+)\b", re.MULTILINE)
_AXIS_TAG_RE = re.compile(
    r"\[lens:\s*(?P<lens>[^|\]]*?)\s*\|\s*lever:\s*(?P<lever>[^|\]]*?)\s*\|"
    r"\s*intent:\s*(?P<intent>[^|\]]*?)\s*\]"
)

REJECT_THRESHOLD = 0.5


@dataclass
class CandidateBlock:
    """One ``## Candidate C-NNN`` section from ``candidates.md``."""

    id: str
    lens: str
    lever: str
    intent: str
    body: str


@dataclass
class CandidateVerdict:
    """kev's retroactive-check score for one candidate."""

    id: str
    lens: str
    lever: str
    intent: str
    pass_prob: float


@dataclass
class CandidateRetrocheck:
    """Result of a retrocheck build: the written file plus per-candidate verdicts."""

    path: Path
    verdicts: list[CandidateVerdict] = field(default_factory=list)
    reject_threshold: float = REJECT_THRESHOLD

    @property
    def rejected_ids(self) -> list[str]:
        return [v.id for v in self.verdicts if v.pass_prob < self.reject_threshold]


def parse_candidates(path: Path) -> list[CandidateBlock]:
    """Split ``candidates.md`` into ``## Candidate C-NNN`` blocks (sync, testable)."""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    headers = list(_CANDIDATE_RE.finditer(text))
    if not headers:
        return []
    blocks: list[CandidateBlock] = []
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        chunk = text[m.end():end]
        tag = _AXIS_TAG_RE.search(chunk)
        blocks.append(
            CandidateBlock(
                id=m.group(1),
                lens=tag.group("lens").strip() if tag else "",
                lever=tag.group("lever").strip() if tag else "",
                intent=tag.group("intent").strip() if tag else "",
                body=chunk.strip(),
            )
        )
    return blocks


def retro_question(intent: str) -> JudgeQ:
    """Map a candidate's intent to its retroactive-check JudgeQ (variant A/B/C)."""
    intent = (intent or "").strip().lower()
    if intent == "preservative-lock":
        return JudgeQ(
            "This candidate argues a passing habit must be preserved (preservative-lock). "
            "If that habit were removed, would the passing trajectories it targets actually "
            "regress and fail?",
            yes_desc="yes: removing it would regress the pass",
            no_desc="no: it would still pass (the lock is not load-bearing)",
        )
    # corrective / preservative-transfer (and any unparsed intent) default to
    # "would the failing trajectory flip to pass".
    return JudgeQ(
        "Given this candidate's proposed change, would the failing trajectories it targets "
        "actually flip to pass? Judge whether the written evidence supports the claim, not "
        "just whether the change sounds plausible.",
        yes_desc="yes: the change would flip the failure to pass",
        no_desc="no: the failure would persist (the change does not actually fix it)",
    )


def _render_state(block: CandidateBlock) -> str:
    return (
        f"candidate {block.id} [lens: {block.lens} | lever: {block.lever} | intent: {block.intent}]\n"
        f"proposal:\n{clip_text(block.body, 1400)}\n"
    )


def _render_markdown(verdicts: list[CandidateVerdict], reject_threshold: float) -> str:
    head = [
        "# Candidate retroactive-check (intent axis)\n",
        "System-1 judgment of each candidate's retroactive check. "
        f"`pass_prob < {reject_threshold}` ⇒ rejected in `--decision-mode enforce`.\n",
        "| candidate | lens | lever | intent | pass_prob | verdict |",
        "|-----------|------|-------|--------|-----------|---------|",
    ]
    body = []
    for v in verdicts:
        verdict = "reject" if v.pass_prob < reject_threshold else "ok"
        body.append(f"| {v.id} | {v.lens} | {v.lever} | {v.intent} | {v.pass_prob:.2f} | {verdict} |")
    return "\n".join(head + body) + "\n"


async def build_candidate_retrocheck(
    *,
    candidates_path: Path,
    output_path: Path,
    model: str = "kev",
    base_url: str = "http://127.0.0.1:8090",
    reject_threshold: float = REJECT_THRESHOLD,
) -> CandidateRetrocheck:
    """Score each candidate's retroactive check through a decision model.

    Returns a :class:`CandidateRetrocheck`; ``rejected_ids`` lists the candidates
    kev judged "no" (``pass_prob < reject_threshold``).  No candidate blocks →
    empty verdicts (the runner treats that as a non-fatal skip).
    """
    blocks = parse_candidates(candidates_path)
    if not blocks:
        _log.warning("no `## Candidate C-N` blocks in %s; retrocheck empty", candidates_path)
        return CandidateRetrocheck(
            path=output_path, verdicts=[], reject_threshold=reject_threshold
        )

    backend = get_decision_model(model, base_url=base_url)
    verdicts: list[CandidateVerdict] = []
    async with backend:
        for b in blocks:
            decisions = await backend.decide(
                state=_render_state(b), questions=[retro_question(b.intent)]
            )
            pass_prob = decisions[0].probabilities.get("true", 0.0)
            verdicts.append(
                CandidateVerdict(
                    id=b.id, lens=b.lens, lever=b.lever, intent=b.intent, pass_prob=pass_prob
                )
            )
            _log.info("retrocheck %s intent=%s pass_prob=%.2f", b.id, b.intent, pass_prob)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_markdown(verdicts, reject_threshold), encoding="utf-8")
    return CandidateRetrocheck(path=output_path, verdicts=verdicts, reject_threshold=reject_threshold)
