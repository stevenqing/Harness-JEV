# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Benchmark-agnostic System-1 trajectory priors for the evolver.

The failure-mode vocabulary is now *meta-induced*.  The strong meta model reads
this round's failed trajectories and proposes K *independent binary* failure
predicates anchored to mechanically-checkable action signatures (e.g. "never
issued ``take <target>``", "issued ``go to <X>`` more than 3 consecutive
times"), then a cheap System-1 read head (kev) scores each failed trajectory
against those predicates as a **noul vector** — K independent p(true) values,
*not* a forced single choice.

This replaces the old abstract 4-way ``lever`` ChooseQ, on which kev's pointer
head collapsed to chance (mean confidence 0.23, "instruction" 37/42): kev
discriminates option *text* that is semantically related to the trajectory
state (goal + actions), not abstract harness-lever names.  Meta-induced
predicates are anchored to exactly that content.

The two *lens* JudgeQs (failure vs. capability-gap) stay benchmark-agnostic and
are prepended to every vector.  What stays per-benchmark is the trajectory
parser/renderer (all benchmarks share the ``write_task_trajectory`` shape:
frontmatter + ``**Extracted action**`` body).

``build_decision_priors`` is the single entry point the runners call right
before ``MetaAgent.evolve()``.  It is a two-role flow: (1) the meta model
*induces* the predicate vocabulary; (2) the configured scoring backend (``kev``
by default) scores every trajectory against that vocabulary as a noul vector.
It returns a :class:`DecisionPriors` (the written file path plus the aggregate
lens/predicate readout).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from harnessx.core.events import Message
from harnessx.decision import JudgeQ
from harnessx.meta_harness.decision_prior import (
    analyze_trajectories,
    render_priors_markdown,
)

_log = logging.getLogger(__name__)


# ── benchmark-agnostic lens axis (看什么) — kept; already a noul JudgeQ ──────
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

LENS_QUESTIONS = [LENS_FAILURE_Q, LENS_CAPABILITY_GAP_Q]
LENS_NAMES = ["lens_failure", "lens_capability_gap"]

# The lever axis is no longer a fixed ChooseQ.  It is induced per-round by the
# meta model (``induce_lever_predicates``) and scored as a noul vector by kev.
DEFAULT_PREDICATE_COUNT = 5

# Fixed hand-written predicate sets (ablation: -induction).  Same {name, q, yes,
# no} shape as the meta-induced predicates, anchored to mechanically-checkable
# action signatures, but static — never re-induced from the round's failures.
FIXED_PREDICATES: dict[str, list[dict[str, str]]] = {
    "alfworld": [
        {
            "name": "never_takes_object",
            "q": "Does the action sequence contain no action beginning with 'take '?",
            "yes": "The agent never issued any take action, so no object was picked up during the trajectory.",
            "no": "The agent issued at least one action beginning with 'take '.",
        },
        {
            "name": "many_invalid_actions",
            "q": "Does the action sequence contain at least 5 actions marked [invalid]?",
            "yes": "At least 5 actions in the sequence are marked [invalid].",
            "no": "Fewer than 5 actions in the sequence are marked [invalid].",
        },
        {
            "name": "excessive_navigation",
            "q": "Does the action sequence contain at least 15 actions beginning with 'go to '?",
            "yes": "The agent issued at least 15 go-to actions, indicating wandering/search thrash.",
            "no": "The agent issued fewer than 15 go-to actions.",
        },
        {
            "name": "observation_loop",
            "q": "Does the action sequence contain 4 or more consecutive actions that are all exactly 'look' or start with 'examine '?",
            "yes": "A run of at least 4 consecutive look/examine actions occurs.",
            "no": "No such run of 4 consecutive look/examine actions occurs.",
        },
        {
            "name": "budget_exhausted",
            "q": "Did the episode end without completing the goal (eval_reason is not 'won') after consuming a large number of steps?",
            "yes": "The episode ran many steps but still did not finish, i.e. it exhausted its budget while blocked.",
            "no": "The episode either completed, or ended early with few steps.",
        },
    ],
}


# ── shared trajectory parser/renderer (write_task_trajectory shape) ─────────
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


# benchmark → (parse, render); the lens questions + induced predicates are universal
_PARSERS: dict[str, tuple[Callable[[Path], dict[str, Any] | None], Callable[[dict[str, Any]], str]]] = {
    "alfworld": (_parse_trajectory, _render_state),
    "gridgames": (_parse_trajectory, _render_state),
    # WebShop / TextCraft / ScienceWorld trajectories all use the same
    # ``write_task_trajectory`` shape (same frontmatter fields + the
    # "**Extracted action**" body), so the shared parser applies unchanged;
    # only the action vocabulary differs (search[]/click[] / get·craft /
    # free-text science actions).
    "webshop": (_parse_trajectory, _render_state),
    "textcraft": (_parse_trajectory, _render_state),
    "sciworld": (_parse_trajectory, _render_state),
}


def _sort_key(path: Path) -> tuple[int, object]:
    """Mirror ``analyze_trajectories``' file ordering so induction and scoring
    read the same trajectories in the same order (deterministic prompt)."""
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


# ── meta induction: K independent binary predicates from failed trajectories ─
_INDUCTION_SYSTEM = (
    "You are analyzing failed agent trajectories to build a failure-diagnostic "
    "vocabulary of independent yes/no predicates."
)

_INDUCTION_TEMPLATE = """Below are {n} failed trajectories. Each is a compact summary: `goal`, exit/eval reason, step count, and the action sequence (with `[invalid]` marking rejected actions).

<trajectories>
{traj}
</trajectories>

Produce exactly {k} INDEPENDENT binary failure predicates — NOT mutually-exclusive classes (one trajectory can satisfy several at once). Each predicate must be:
- phrased as a yes/no question;
- its "yes" anchored to a mechanically-checkable signature in the action sequence (e.g. "never issued take <target>", "issued go to <X> more than 3 consecutive times", "issued put <obj> into a wrong receptacle", "repeated the same examine/look without progress");
- its "no" is the negation.

Return ONLY a JSON object mapping a short predicate name -> {{"q": "<yes/no question>", "yes": "<what yes means>", "no": "<what no means>"}}. No prose, no markdown fences."""


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a generative reply (fence-tolerant)."""
    text = (text or "").strip()
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


async def induce_lever_predicates(
    provider: Any,
    states: list[str],
    k: int = DEFAULT_PREDICATE_COUNT,
) -> list[dict]:
    """Ask the meta model for K independent binary predicates covering ``states``.

    Returns ``[{name, q, yes, no}, ...]`` in meta's own order.  Retries a few
    times because the mioffice gateway can return an empty text block on a
    transient; falls back to the thinking block when the text block is empty.
    """
    if not states:
        return []  # nothing to induce from; the caller handles an empty vector
    traj = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(states))
    prompt = _INDUCTION_TEMPLATE.format(n=len(states), k=k, traj=traj)
    for attempt in range(4):
        try:
            event = await provider.complete(
                messages=[
                    Message(role="system", content=_INDUCTION_SYSTEM),
                    Message(role="user", content=prompt),
                ],
                tools=[],
            )
            obj = _extract_json(event.content) or _extract_json(event.thinking)
            if obj:
                preds = []
                for name, v in obj.items():
                    if not isinstance(v, dict) or not all(kk in v for kk in ("q", "yes", "no")):
                        raise ValueError(f"malformed predicate {name!r}: {v!r}")
                    preds.append(
                        {"name": str(name), "q": str(v["q"]), "yes": str(v["yes"]), "no": str(v["no"])}
                    )
                return preds
        except Exception as exc:  # noqa: BLE001 — retry transient gateway faults / malformed JSON
            _log.warning("meta predicate induction attempt %d failed: %s", attempt + 1, exc)
        await asyncio.sleep(8 * (attempt + 1))
    raise RuntimeError("meta failed to induce lever predicates after 4 attempts")


def _inject_predicate_vocab(
    body: str, predicates: list[dict], inducing_seconds: float, source: str = "meta", scorer: str = "kev"
) -> str:
    """Prepend a human-readable definition of each predicate to the priors markdown.

    ``source`` selects the provenance wording ("meta" induced each round vs
    "fixed" static hand-written), and ``scorer`` is the backend that scores the
    noul vector ("kev" vs "meta").
    """
    if not predicates:
        return body
    if source == "fixed":
        header = "## Fixed failure predicates (hand-written, noul vector)"
        provenance = (
            f"{len(predicates)} static hand-written binary predicates (no meta induction "
            f"this round). Each trajectory is scored p(true) per predicate by the {scorer} "
            "backend — these are PRIORS, not labels."
        )
    else:
        header = "## Induced failure predicates (meta-induced, noul vector)"
        provenance = (
            f"{len(predicates)} independent binary predicates induced by the meta model from "
            f"this round's failures (induction wall-clock {inducing_seconds:.1f}s). Each "
            f"trajectory is scored p(true) per predicate by the {scorer} backend — these are "
            "PRIORS, not labels."
        )
    lines = [header, "", provenance]
    for i, p in enumerate(predicates, 1):
        lines.append(f"{i}. **{p['name']}** — *{p['q']}*  (yes = {p['yes']}; no = {p['no']})")
    block = "\n".join(lines) + "\n\n"
    return body.replace("# System-1 trajectory priors\n\n", "# System-1 trajectory priors\n\n" + block, 1)


@dataclass
class DecisionPriors:
    """Result of a prior build: the written file plus the aggregate readout."""

    path: Path
    lever_predicates: list[dict]       # [{name, q, yes, no}] — the meta-induced vocabulary
    lever_aggregate: dict[str, float]  # predicate name -> round-level mean p(true)
    lens_failure_p: float
    lens_capability_gap_p: float
    judging_seconds: float
    inducing_seconds: float


async def build_decision_priors(
    *,
    benchmark: str,
    trajectories_dir: Path,
    output_path: Path,
    model: str = "kev",
    base_url: str = "http://127.0.0.1:8090",
    max_trajectories: int = 96,
    model_kwargs: dict | None = None,
    meta_provider: Any = None,
    predicate_count: int = DEFAULT_PREDICATE_COUNT,
    predicate_source: str = "meta",
    concurrency: int = 1,
    constant_prior: float | None = None,
) -> DecisionPriors:
    """Render meta-induced predicate priors for a benchmark's failed trajectories.

    Two roles: the meta model (``meta_provider``) *induces* ``predicate_count``
    independent binary failure predicates from this round's failures; the scoring
    backend (``model``, default ``kev``) then scores every trajectory against
    those predicates as a noul vector.  The two lens JudgeQs are prepended and
    stay fixed.

    ``predicate_source`` selects the lever vocabulary for the ablation study:
    ``"meta"`` (default) induces it each round, ``"fixed"`` uses
    :data:`FIXED_PREDICATES` for the benchmark (static, no meta call), and
    ``"none"`` emits no lever predicates (lens-only).

    ``constant_prior`` (control experiment) is an optional scalar that, when set,
    overwrites every decision with a fixed p(true)=``constant_prior`` after the
    scoring pass.  This keeps the *structure* of the prior (the SOUL step-3
    machine-rendered table + aggregate row) while erasing the *values*, isolating
    "does the prior structure help" from "do the base-rate values help".
    """
    try:
        parse_trajectory, render_state = _PARSERS[benchmark]
    except KeyError:
        raise NotImplementedError(
            f"decision priors not wired for benchmark {benchmark!r}; available: {sorted(_PARSERS)}"
        ) from None

    if meta_provider is None:
        raise ValueError(
            "build_decision_priors requires `meta_provider` — the meta model induces "
            "the predicate vocabulary before the scoring backend scores it"
        )

    # 1. collect the failed states (one light pass, for induction only)
    files = sorted(trajectories_dir.glob("*.md"), key=_sort_key)[:max_trajectories]
    states: list[str] = []
    for path in files:
        fields = parse_trajectory(path)
        if fields is not None:
            states.append(render_state(fields))

    # 2. obtain the predicate vocabulary (meta-induced / fixed / none)
    if predicate_source == "fixed":
        predicates = [dict(p) for p in FIXED_PREDICATES.get(benchmark, [])]
        inducing_seconds = 0.0
    elif predicate_source == "none":
        predicates = []
        inducing_seconds = 0.0
    else:  # "meta" (default)
        started = time.perf_counter()
        predicates = await induce_lever_predicates(meta_provider, states, k=predicate_count)
        inducing_seconds = time.perf_counter() - started

    # 3. assemble questions: 2 lens + K induced predicates (all noul)
    lever_questions = [
        JudgeQ(instructions=p["q"], yes_desc=p["yes"], no_desc=p["no"]) for p in predicates
    ]
    questions = [*LENS_QUESTIONS, *lever_questions]
    names = [*LENS_NAMES, *[p["name"] for p in predicates]]

    # 4. score every trajectory against the assembled questions
    kwargs = dict(model_kwargs or {})
    kwargs.setdefault("base_url", base_url)
    rows, judging_seconds = await analyze_trajectories(
        trajectories_dir=trajectories_dir,
        questions=questions,
        parse_trajectory=parse_trajectory,
        render_state=render_state,
        model=model,
        model_kwargs=kwargs,
        max_trajectories=max_trajectories,
        concurrency=concurrency,
    )
    if not rows:
        _log.warning("no failed trajectories to analyze under %s; priors will be empty", trajectories_dir)

    # 4b. (control experiment) erase the scored values, keep the structure.
    if constant_prior is not None:
        c = float(constant_prior)
        for row in rows:
            for dec in row.decisions:
                dec.probabilities = {"false": 1.0 - c, "true": c}
                dec.argmax = "true" if c >= 0.5 else "false"
                dec.confidence = abs(c - 0.5) * 2.0

    # 5. render markdown (predicate definitions + noul table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = render_priors_markdown(
        rows=rows, questions=questions, names=names, judging_seconds=judging_seconds
    )
    output_path.write_text(
        _inject_predicate_vocab(body, predicates, inducing_seconds, source=predicate_source, scorer=model),
        encoding="utf-8",
    )

    # 6. aggregate the round-level readout
    n = len(rows)
    if rows:
        lens_failure = sum(r.decisions[0].probabilities["true"] for r in rows) / n
        lens_capability = sum(r.decisions[1].probabilities["true"] for r in rows) / n
        lever_aggregate = {
            p["name"]: sum(r.decisions[2 + i].probabilities["true"] for r in rows) / n
            for i, p in enumerate(predicates)
        }
    else:
        lens_failure = lens_capability = 0.0
        lever_aggregate = {p["name"]: 0.0 for p in predicates}

    return DecisionPriors(
        path=output_path,
        lever_predicates=predicates,
        lever_aggregate=lever_aggregate,
        lens_failure_p=lens_failure,
        lens_capability_gap_p=lens_capability,
        judging_seconds=judging_seconds,
        inducing_seconds=inducing_seconds,
    )
