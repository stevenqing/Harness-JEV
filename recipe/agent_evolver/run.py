# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""ALFWorld / WebShop evolver: text-protocol harness evolution.

Usage::

    # Re-measure the baseline (R0 only, no evolve)
    python -m recipe.agent_evolver.run alfworld --num-rounds 1 --num-tasks 64
    python -m recipe.agent_evolver.run webshop  --num-rounds 1 --num-tasks 64

    # One evolve round (R0 baseline rollout → meta-agent evolves → R1 rollout)
    python -m recipe.agent_evolver.run alfworld --num-rounds 2 --num-tasks 64 --run-tag smoke

    # Held-out split (separate env servers / model generalization probe)
    python -m recipe.agent_evolver.run alfworld --split heldout --num-rounds 3

How it works
------------
The agent is Qwen3.5-4B driven by the Skill1 ``<think>/<action>`` ReAct loop
(``agents.run_alfworld`` / ``agents.run_webshop``) — the same loop that yields
the 0.344 / 0.281 baselines.  The evolvable artifact is a ``HarnessConfig``
that *carries* the text-loop spec (system prompt + ``guidance`` + loop-processor
params); each round ``meta_agent.evolve()`` edits that YAML, and the next round
rolls out with the decoded spec.

Model weights are never modified — only the system prompt / guidance / loop
params evolve.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── load .env ────────────────────────────────────────────────────────────────
_env_path = Path(_PROJECT_ROOT) / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# Stop LiteLLM from fetching the remote model cost map at startup (no egress on
# this host); must be set BEFORE `import litellm`.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm as _litellm

_litellm.suppress_debug_info = True

from harnessx.meta_harness import MetaAgent
from harnessx.core.harness import HarnessConfig
from harnessx.core.model_config import ModelConfig
from harnessx.providers.anthropic_provider import AnthropicProvider
from harnessx.providers.litellm_provider import LiteLLMProvider

from .agents import run_alfworld, run_webshop
from .gate import score_and_gate
from .native.config import (
    make_alfworld_baseline_config,
    make_webshop_baseline_config,
    spec_from_config,
)
from .spec import ALF_MAX_STEPS, AGENT_ENABLE_THINKING, AGENT_MAX_TOKENS, AGENT_MODEL, AGENT_TEMPERATURE, AGENT_TOP_P, WS_MAX_STEPS
from .trajectories import write_task_trajectory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)
for _lg in ("LiteLLM", "httpx", "httpx2", "litellm", "litellm.litellm_core_utils"):
    logging.getLogger(_lg).setLevel(logging.WARNING)
logger = logging.getLogger("agent_evolver")

_RECIPE_DIR = Path(__file__).resolve().parent
RUNS_DIR = _RECIPE_DIR / "runs"

# ── Defaults (env-overridable, aligned to the Skill1 / verl-agent base) ──────
_AGENT_API_BASES_DEFAULT = os.environ.get(
    "EVOLVER_AGENT_API_BASES",
    "http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,"
    "http://127.0.0.1:8202/v1,http://127.0.0.1:8203/v1",
)
_ALF_HELDIN_ENV_URLS = (
    "http://127.0.0.1:18082,http://127.0.0.1:18083,"
    "http://127.0.0.1:18084,http://127.0.0.1:18085"
)
_ALF_HELDOUT_ENV_URLS = (
    "http://127.0.0.1:18086,http://127.0.0.1:18087,"
    "http://127.0.0.1:18088,http://127.0.0.1:18089"
)
_WS_ENV_URL = "http://127.0.0.1:18090"

DEFAULT_META_MODEL = os.environ.get("EVOLVER_META_MODEL", "anthropic/volcengine_maas/deepseek-v4-pro")
DEFAULT_META_API_BASE = os.environ.get(
    "EVOLVER_META_API_BASE", os.environ.get("ANTHROPIC_BASE_URL", "https://api.llm.mioffice.cn/anthropic")
)

NUM_ROUNDS = 3
MAX_TASKS = 64
EVOLVE_COST_CAP_USD = 50.0
EVOLVE_MAX_STEPS = 100
EVOLVE_WALL_CLOCK_S = 3600
REGRESSION_TOLERANCE = 0.02
COST_WEIGHT = 0.0

# Benchmark-specific playbook skills for the meta-agent.
_SKILLS_DIR = _RECIPE_DIR / "skills"


# ── Providers ─────────────────────────────────────────────────────────────────

def make_agent_provider(api_base: str) -> LiteLLMProvider:
    """Weak agent model (Qwen3.5-4B) — S1 decoding (thinking on, 4096, temp 0.4)."""
    return LiteLLMProvider(
        model=AGENT_MODEL,
        api_base=api_base,
        api_key="EMPTY",
        temperature=AGENT_TEMPERATURE,
        top_p=AGENT_TOP_P,
        max_tokens=AGENT_MAX_TOKENS,
        extra_body={"chat_template_kwargs": {"enable_thinking": AGENT_ENABLE_THINKING}},
    )


def _make_provider(model: str, api_base: str | None, *, max_tokens: int = 16384):
    """Meta-agent provider (mirrors tau2): anthropic/* → AnthropicProvider, else LiteLLM.

    NOTE: extended thinking is intentionally OFF.  The mioffice deepseek fork's
    ``AsyncMessages.create`` has no ``temperature`` param, and ``AnthropicProvider``
    sets ``temperature=1`` when ``extended_thinking`` is on → TypeError.  The fork
    emits thinking blocks natively, so we just give it a large ``max_tokens``
    (≥4096 or thinking eats the budget).
    """
    if model.startswith("anthropic/"):
        model_name = model[len("anthropic/"):]
        return AnthropicProvider(
            model=model_name,
            base_url=api_base,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=max_tokens,
        )
    # Non-anthropic: assume an OpenAI-compatible endpoint — e.g. a locally-served
    # preset model on multi-machine pods, where the anthropic gateway is
    # unreachable.  Forward api_base + api_key exactly like make_agent_provider
    # does; the old LiteLLM branch dropped api_base, so --meta-api-base was
    # silently ignored for non-anthropic models.
    return LiteLLMProvider(
        model,
        api_base=api_base,
        api_key=os.environ.get("ANTHROPIC_API_KEY", "EMPTY"),
        max_tokens=max_tokens,
    )


# ── Rollout ───────────────────────────────────────────────────────────────────

def _shard(items: list, n: int) -> list[list]:
    per = (len(items) + n - 1) // n
    return [items[i * per:(i + 1) * per] for i in range(n)]


async def run_text(
    config: HarnessConfig,
    benchmark: str,
    task_ids: list[int],
    api_bases: list[str],
    env_urls: list[str],
    seed: int,
    concurrency: int,
) -> list[dict]:
    """Roll out the text-loop spec decoded from ``config`` over ``task_ids``.

    One worker per agent api_base, each serially processing a contiguous shard
    against ``env_urls[i % len(env_urls)]``.  ALFWorld env servers are
    single-instance/stateful, so a worker must not overlap episodes on one
    server; WebShop creates a fresh uuid instance per task so one server can
    absorb several workers.
    """
    spec = spec_from_config(config)
    n_workers = max(1, min(len(api_bases), concurrency))
    shards = _shard(task_ids, n_workers)

    async def worker(i: int) -> list[dict]:
        api_base = api_bases[i % len(api_bases)]
        env_url = env_urls[i % len(env_urls)]
        provider = make_agent_provider(api_base)
        if benchmark == "alfworld":
            return await run_alfworld(provider, spec, shards[i], seed=seed, env_url=env_url)
        return await run_webshop(provider, spec, shards[i], seed=seed, env_url=env_url)

    chunks = await asyncio.gather(*(worker(i) for i in range(n_workers)))
    return [r for chunk in chunks for r in chunk]


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmark", choices=["alfworld", "webshop"])
    p.add_argument("--split", choices=["heldin", "heldout"], default="heldin",
                   help="ALFWorld env split (heldin 18082-18085 / heldout 18086-18089). Ignored for webshop.")
    p.add_argument("--num-tasks", type=int, default=MAX_TASKS)
    p.add_argument("--start", type=int, default=0, help="task index offset (game_idx / task_id)")
    p.add_argument("--seed", type=int, default=None, help="defaults to 1234 (alfworld) / 0 (webshop)")
    p.add_argument("--num-rounds", type=int, default=NUM_ROUNDS,
                   help="total rollout rounds (R0..R{N-1}); evolve runs after each non-last round")
    p.add_argument("--agent-model", default=AGENT_MODEL)
    p.add_argument("--agent-api-bases", default=None, help="comma-separated vLLM /v1 bases")
    p.add_argument("--env-urls", default=None, help="comma-separated env server bases (override default)")
    p.add_argument("--concurrency", type=int, default=None,
                   help="max concurrent tasks (default = len(agent-api-bases))")
    p.add_argument("--meta-model", default=DEFAULT_META_MODEL)
    p.add_argument("--meta-api-base", default=DEFAULT_META_API_BASE)
    p.add_argument("--evolve-cost", type=float, default=EVOLVE_COST_CAP_USD)
    p.add_argument("--evolve-steps", type=int, default=EVOLVE_MAX_STEPS)
    p.add_argument("--evolve-wall-clock", type=int, default=EVOLVE_WALL_CLOCK_S)
    p.add_argument("--regression-tolerance", type=float, default=REGRESSION_TOLERANCE)
    p.add_argument("--cost-weight", type=float, default=COST_WEIGHT)
    p.add_argument("--run-tag", default=None, help="output tag; defaults to <benchmark>_<split>_<timestamp>")
    p.add_argument("--base-config", default=None,
                   help="start from an external config YAML (e.g. a held-in evolved config) "
                        "instead of the programmatic baseline — for held-out generalization probes")
    p.add_argument("--decision-backend", default="llm",
                   help="trajectory-analysis backend for the meta-evolve decision process: "
                        "'llm' (default — the meta-LLM does all diagnosis itself), 'rule' "
                        "(the deterministic A2 rule tagger), or a registered jevlike "
                        "decision-model name (e.g. 'kev') to inject System-1 priors into "
                        "the brief. Unknown/offline model backends fall back to 'llm'.")
    p.add_argument("--decision-base-url", default="http://127.0.0.1:8090",
                   help="decision-model sidecar base URL (used when --decision-backend != llm)")
    p.add_argument("--decision-mode", default="prior", choices=("prior", "enforce"),
                   help="how to use the decision backend: 'prior' (soft — inject as an "
                        "overrideable prior, default) or 'enforce' (hard — the backend's lever "
                        "decision is binding and gate-checked against the authored config)")
    return p.parse_args(argv)


def _cited_rejected(retro: Any, memo_path: Path) -> list[str]:
    """Retrocheck gate: reject the *cited* candidates kev judged "no".

    When the latest journal entry's ``cited_candidates`` frontmatter is
    readable, only a cited candidate may reject the round (a weak un-cited
    candidate must not sink a good round).  Unreadable / absent → gate on all
    candidates (conservative).
    """
    if not retro.verdicts:
        return []
    rejected = {v.id for v in retro.verdicts if v.id in retro.rejected_ids}
    try:
        from harnessx.meta_harness import journal as _journal

        latest = _journal.latest_entry(str(memo_path))
        if latest is not None:
            raw = latest.frontmatter.get("cited_candidates")
            if isinstance(raw, list):
                cited = {str(c) for c in raw}
            elif isinstance(raw, str):
                cited = {raw}
            else:
                cited = None
            if cited is not None:
                rejected &= cited
    except Exception:  # noqa: BLE001 — best-effort; fall back to all-rejected
        pass
    return sorted(rejected)


async def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    benchmark = args.benchmark
    split = args.split if benchmark == "alfworld" else "test"

    api_bases = [u.strip() for u in (args.agent_api_bases or _AGENT_API_BASES_DEFAULT).split(",") if u.strip()]
    concurrency = args.concurrency or len(api_bases)

    if benchmark == "alfworld":
        default_env = _ALF_HELDIN_ENV_URLS if split == "heldin" else _ALF_HELDOUT_ENV_URLS
    else:
        default_env = _WS_ENV_URL
    env_urls = [u.strip() for u in (args.env_urls or default_env).split(",") if u.strip()]

    seed = args.seed if args.seed is not None else (1234 if benchmark == "alfworld" else 0)
    max_steps = ALF_MAX_STEPS if benchmark == "alfworld" else WS_MAX_STEPS
    task_ids = list(range(args.start, args.start + args.num_tasks))

    run_tag = args.run_tag or f"{benchmark}_{split}_{time.strftime('%Y%m%d_%H%M%S')}"
    RUN_DIR = RUNS_DIR / "evolve" / run_tag
    LEARNINGS_PATH = RUN_DIR / "learnings.md"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("run_tag=%s benchmark=%s split=%s tasks=%d rounds=%d", run_tag, benchmark, split, len(task_ids), args.num_rounds)
    logger.info("api_bases=%s env_urls=%s seed=%d concurrency=%d", api_bases, env_urls, seed, concurrency)

    # ── Meta-agent (one per run; persistent model/memo/budgets) ───────────
    meta_provider = _make_provider(args.meta_model, args.meta_api_base)
    meta_model = ModelConfig(main=meta_provider)
    meta_agent = MetaAgent(
        inner_model=meta_model,
        memo_path=str(LEARNINGS_PATH),
        extra_skills_dirs=([_SKILLS_DIR] if _SKILLS_DIR.is_dir() else None),
        max_cost_usd=args.evolve_cost,
        wall_clock_s=float(args.evolve_wall_clock),
        max_steps=args.evolve_steps,
    )

    # ── Baseline config (programmatic, relocatable absolute paths) ─────────
    if args.base_config:
        current_config = HarnessConfig.from_yaml_file(args.base_config)
        logger.info("starting from external config: %s", args.base_config)
    elif benchmark == "alfworld":
        current_config = make_alfworld_baseline_config()
    else:
        current_config = make_webshop_baseline_config()
    current_config = current_config.canonicalize()

    # ── Pre-flight: fail fast if enforce mode's decision backend is down ──
    if args.decision_backend != "llm" and args.decision_mode == "enforce":
        from harnessx.decision import JudgeQ, get_decision_model

        try:
            backend = get_decision_model(args.decision_backend, base_url=args.decision_base_url)
            async with backend:
                await backend.decide(state="preflight", questions=[JudgeQ("is the sidecar up?")])
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "enforce mode: decision backend %r unreachable at %s — aborting before rollout: %s",
                args.decision_backend, args.decision_base_url, exc,
            )
            raise SystemExit(1) from exc
        logger.info("decision backend %r reachable at %s", args.decision_backend, args.decision_base_url)

    best_so_far: tuple[float, float, Any, int] | None = None
    next_evolve_status: str = "baseline"
    all_rounds: list[list[dict]] = []
    round_summaries: list[dict] = []

    for round_idx in range(args.num_rounds):
        is_last = round_idx == args.num_rounds - 1

        round_dir = RUN_DIR / f"R{round_idx}"
        traj_dir = round_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)

        round_config_path = round_dir / "config.yaml"
        current_config.to_yaml_file(round_config_path)

        config_label = "baseline" if round_idx == 0 else f"evolved_R{round_idx}"
        logger.info("\n" + "=" * 60)
        logger.info("ROUND %d/%d  [%s]", round_idx, args.num_rounds - 1, config_label)
        logger.info("=" * 60)

        # ── Rollout ─────────────────────────────────────────────────────────
        records = await run_text(current_config, benchmark, task_ids, api_bases, env_urls, seed, concurrency)
        for rec in records:
            write_task_trajectory(traj_dir, rec, max_steps)
            rec.pop("trajectory", None)  # keep memory light; trajectory is on disk
        gc.collect()
        all_rounds.append(records)

        passed = sum(1 for r in records if r["success"])
        avg_reward = sum(r["reward"] for r in records) / len(records) if records else 0.0
        round_cost = 0.0  # text loop has no cost accounting; cost_weight defaults to 0
        logger.info("[R%d] pass=%d/%d (%.3f)  mean_reward=%.4f",
                    round_idx, passed, len(records), (passed / len(records)) if records else 0.0,
                    avg_reward)
        round_summaries.append({
            "round": round_idx, "config": config_label, "tasks": len(records),
            "passed": passed, "pass_rate": round((passed / len(records)), 3) if records else 0.0,
            "mean_reward": round(avg_reward, 4),
        })

        # ── Gate ─────────────────────────────────────────────────────────────
        decision, reason, best_so_far, reverted_cfg = score_and_gate(
            round_reward=avg_reward,
            round_cost=round_cost,
            round_idx=round_idx,
            round_config=current_config,
            best=best_so_far,
            tolerance=args.regression_tolerance,
            cost_weight=args.cost_weight,
        )
        logger.info("[R%d] gate: %s — %s", round_idx, decision, reason)
        if reverted_cfg is not None:
            logger.warning("[R%d] REGRESSION — reverting current_config to R%d", round_idx, best_so_far[3])
            current_config = reverted_cfg

        # ── Journal back-fill (orchestrator contract, R1+) ──────────────────
        if round_idx >= 1:
            try:
                from harnessx.meta_harness import journal as _journal

                entries = _journal.read_entries(LEARNINGS_PATH)
                entry = next((e for e in entries if e.round == round_idx), None)
                if entry is not None:
                    prev_records = all_rounds[-2] if len(all_rounds) >= 2 else []
                    prev_passed = {str(r["task_id"]) for r in prev_records if r["success"]}
                    prev_appeared = {str(r["task_id"]) for r in prev_records}
                    cur_passed = {str(r["task_id"]) for r in records if r["success"]}
                    cur_appeared = {str(r["task_id"]) for r in records}
                    outcome = "reverted" if decision == "reject" else "accepted"
                    if next_evolve_status == "noop":
                        outcome = "noop"
                    elif next_evolve_status == "rejected_retrocheck":
                        outcome = "reverted"
                    attribution = _journal.compute_attribution(
                        entry.predicted_affected,
                        passed_now=cur_passed,
                        passed_before=prev_passed,
                        appeared_now=cur_appeared,
                        appeared_before=prev_appeared,
                    )
                    predicted_set = set(entry.predicted_affected)
                    regressed_unpredicted = sorted(
                        (prev_passed & prev_appeared & cur_appeared) - cur_passed - predicted_set
                    )
                    _journal.fill_gating(
                        LEARNINGS_PATH, round_idx, outcome, attribution,
                        extra_frontmatter={"regressed_unpredicted": regressed_unpredicted},
                    )
                    logger.info("[R%d] journal fill_gating outcome=%s attribution=%s", round_idx, outcome, attribution)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[R%d] journal fill_gating failed (non-fatal): %s", round_idx, exc)

        if is_last:
            continue

        # ── Evolve: produce next round's config ─────────────────────────────
        next_round_dir = RUN_DIR / f"R{round_idx + 1}"
        evolve_dir = next_round_dir / "evolve"
        logger.info("[R%d] evolve → %s (memo=%s)", round_idx, evolve_dir, LEARNINGS_PATH)

        existing_yaml = evolve_dir / "config.yaml"
        if existing_yaml.is_file():
            candidate = HarnessConfig.from_yaml_file(existing_yaml).canonicalize()
            next_evolve_status = "ok"
            current_config = candidate
            logger.info("[R%d] evolve skipped — using existing %s", round_idx, existing_yaml)
        else:
            priors_path = None
            decision_lever = None
            decision_lever_confidence = None
            if args.decision_backend != "llm":
                try:
                    priors_path = evolve_dir / "_meta_scratch" / "decision_priors.md"
                    if args.decision_backend == "rule":
                        # A3 rule arm: inject the deterministic A2 rule tagger in
                        # the same brief channel as the kev prior (no model call).
                        from .rule_prior import build_rule_priors

                        build_rule_priors(
                            benchmark=benchmark,
                            trajectories_dir=traj_dir,
                            output_path=priors_path,
                        )
                        logger.info("[R%d] rule-tagger priors → %s", round_idx, priors_path)
                    else:
                        from .decision_prior import build_decision_priors

                        priors = await build_decision_priors(
                            benchmark=benchmark,
                            trajectories_dir=traj_dir,
                            output_path=priors_path,
                            model=args.decision_backend,
                            base_url=args.decision_base_url,
                        )
                        priors_path = priors.path
                        logger.info("[R%d] decision priors (%s) → %s (lever=%s conf=%.2f)",
                                    round_idx, args.decision_backend, priors_path,
                                    priors.lever_argmax, priors.lever_confidence)
                        if args.decision_mode == "enforce" and priors.lever_argmax is not None:
                            decision_lever = priors.lever_argmax
                            decision_lever_confidence = priors.lever_confidence
                        elif args.decision_mode == "enforce":
                            logger.warning(
                                "[R%d] enforce mode but no failed trajectories to bind a lever "
                                "(lever_argmax is None) — running unbound this round",
                                round_idx,
                            )
                except Exception as exc:  # noqa: BLE001
                    if args.decision_mode == "enforce":
                        # In enforce mode a silent fallback would defeat the experiment:
                        # the lever would go unbound and the retrocheck would be skipped,
                        # turning the arm into a plain prior/llm arm. Fail loud instead.
                        logger.error(
                            "[R%d] enforce mode: decision backend %s unreachable (%s) — "
                            "cannot bind lever or retrocheck; aborting",
                            round_idx, args.decision_backend, exc,
                        )
                        raise
                    logger.warning("[R%d] decision priors skipped (non-fatal): %s", round_idx, exc)
                    priors_path = None
            pre_evolve_config = current_config
            try:
                new_yaml = await meta_agent.evolve(
                    current_config=round_config_path,
                    trajectories_dir=traj_dir,
                    output_dir=evolve_dir,
                    decision_priors_path=priors_path,
                    decision_lever=decision_lever,
                    decision_lever_confidence=decision_lever_confidence,
                )
                candidate = HarnessConfig.from_yaml_file(new_yaml).canonicalize()
                next_evolve_status = "noop" if round_config_path.read_bytes() == Path(new_yaml).read_bytes() else "ok"
                logger.info("[R%d] → R%d config: %s (status=%s)", round_idx, round_idx + 1, new_yaml, next_evolve_status)
                current_config = candidate
            except Exception as exc:  # noqa: BLE001
                next_evolve_status = "crashed"
                logger.exception("[R%d] evolve crashed — R%d reuses current config: %s", round_idx, round_idx + 1, exc)

            # ── Phase 3: candidate retroactive-check (intent axis, enforce) ──
            if next_evolve_status == "ok" and args.decision_mode == "enforce" and args.decision_backend != "llm":
                try:
                    from .candidate_retrocheck import build_candidate_retrocheck

                    retro = await build_candidate_retrocheck(
                        candidates_path=evolve_dir / "_meta_scratch" / "candidates.md",
                        output_path=evolve_dir / "_meta_scratch" / "candidate_retrocheck.md",
                        model=args.decision_backend,
                        base_url=args.decision_base_url,
                    )
                    rejected = _cited_rejected(retro, LEARNINGS_PATH)
                    if rejected:
                        next_evolve_status = "rejected_retrocheck"
                        current_config = pre_evolve_config
                        logger.warning(
                            "[R%d] retrocheck rejected %s — R%d reuses current config",
                            round_idx, rejected, round_idx + 1,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[R%d] retrocheck skipped (non-fatal): %s", round_idx, exc)

        # Cooldown: the meta-agent bursts many requests; wait out rate-limit window.
        await asyncio.sleep(60)

    # ── Final report ─────────────────────────────────────────────────────────
    summary = {
        "benchmark": benchmark,
        "split": split,
        "run_tag": run_tag,
        "agent_model": args.agent_model,
        "seed": seed,
        "num_tasks": len(task_ids),
        "rounds": round_summaries,
    }
    (RUN_DIR / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for s in round_summaries:
        logger.info("summary: R%(round)d %(config)s pass=%(passed)d/%(tasks)d rate=%(pass_rate)s "
                    "mean_reward=%(mean_reward)s", s)
    logger.info("done → %s", RUN_DIR)


if __name__ == "__main__":
    asyncio.run(main())
