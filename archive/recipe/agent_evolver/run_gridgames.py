# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Gridgame (FrozenLake / Sokoban) evolver — Phase C: harness evolution.

Usage::

    # Baseline only (R0 rollout, no evolve)
    python -m recipe.agent_evolver.run_gridgames --game frozenlake --tier L4 --num-rounds 1 --num-tasks 8

    # One evolve round (R0 rollout → meta-agent evolves → R1 rollout)
    python -m recipe.agent_evolver.run_gridgames --game frozenlake --tier L4 --num-rounds 2 --num-tasks 8 --run-tag smoke

    # Formal 6-round evolution on the full FL L4 evolve split (256 levels)
    python -m recipe.agent_evolver.run_gridgames --game frozenlake --tier L4 --num-rounds 6 --run-tag fl_l4_r6

    # Held-out generalization probe (start from an evolved config)
    python -m recipe.agent_evolver.run_gridgames --game frozenlake --tier L4 --split test \
        --num-rounds 1 --base-config <R6/config.yaml>

How it works
------------
The agent is Qwen3.5-4B driven by the R0 loop of spec section 5 (one model call
per move, a fixed action parser).  The evolvable artifact is a ``HarnessConfig``
that *carries* the ``GridgameSpec`` (system prompt + guidance + history window);
each round ``meta_agent.evolve()`` edits that YAML, and the next round rolls out
with the decoded spec.  Decoding is spec §5 (thinking on, max_tokens 4096).

Model weights are never modified — only the system prompt / guidance / history
window evolve.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
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

import httpx
from harnessx.meta_harness import MetaAgent
from harnessx.core.harness import HarnessConfig
from harnessx.core.model_config import ModelConfig
from harnessx.providers.anthropic_provider import AnthropicProvider
from harnessx.providers.litellm_provider import LiteLLMProvider

from .gate import score_and_gate
from .gridgame_evolve import filter_levels, load_levels, run_gridgame_spec
from .native.gridgame_config import make_gridgame_baseline_config, gridgame_spec_from_config
from .trajectories import write_task_trajectory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)
for _lg in ("LiteLLM", "httpx", "litellm", "litellm.litellm_core_utils"):
    logging.getLogger(_lg).setLevel(logging.WARNING)
logger = logging.getLogger("agent_evolver")

_RECIPE_DIR = Path(__file__).resolve().parent
RUNS_DIR = _RECIPE_DIR / "runs"

# ── Defaults ─────────────────────────────────────────────────────────────────
AGENT_MODEL = os.environ.get("EVOLVER_AGENT_MODEL", "Qwen3.5-4B")
AGENT_API_BASE = os.environ.get("EVOLVER_AGENT_API_BASE", "http://127.0.0.1:8200/v1")

DEFAULT_META_MODEL = os.environ.get("EVOLVER_META_MODEL", "anthropic/volcengine_maas/deepseek-v4-pro")
DEFAULT_META_API_BASE = os.environ.get(
    "EVOLVER_META_API_BASE", os.environ.get("ANTHROPIC_BASE_URL", "https://api.llm.mioffice.cn/anthropic")
)

NUM_ROUNDS = 6
MAX_TASKS = 0  # 0 = all levels in the cell's split (thinking-off decoding is fast)
EVOLVE_COST_CAP_USD = 50.0
EVOLVE_MAX_STEPS = 100
EVOLVE_WALL_CLOCK_S = 3600
REGRESSION_TOLERANCE = 0.02
COST_WEIGHT = 0.0

# Benchmark-specific playbook skills for the meta-agent.
_SKILLS_DIR = _RECIPE_DIR / "skills"


# ── Providers ─────────────────────────────────────────────────────────────────

def _make_provider(model: str, api_base: str | None, *, max_tokens: int = 16384):
    """Meta-agent provider (mirrors run.py / tau2): anthropic/* → AnthropicProvider.

    NOTE: extended thinking is intentionally OFF.  The mioffice deepseek fork's
    ``AsyncMessages.create`` has no ``temperature`` param, and ``AnthropicProvider``
    sets ``temperature=1`` when ``extended_thinking`` is on → TypeError.  The fork
    emits thinking blocks natively, so we give it a large ``max_tokens`` (≥4096).
    """
    if model.startswith("anthropic/"):
        model_name = model[len("anthropic/"):]
        return AnthropicProvider(
            model=model_name,
            base_url=api_base,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=max_tokens,
        )
    return LiteLLMProvider(model, extra_headers={"X-Model-Provider-Id": "YOUR_PROVIDER_ID"})


def _cited_rejected(retro: Any, memo_path: Path) -> list[str]:
    """Retrocheck gate: reject the *cited* candidates kev judged "no".

    Mirrors run.py: when the latest journal entry's ``cited_candidates``
    frontmatter is readable, only a cited candidate may reject the round.
    Unreadable / absent → gate on all candidates (conservative).
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


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", choices=["frozenlake", "sokoban"], default="frozenlake")
    p.add_argument("--harness", choices=["bracket", "ragen"], default="bracket",
                   help="starting harness protocol: bracket (R0 '[up]') or ragen "
                        "(<think>/<answer> + thinking-off + short budget)")
    p.add_argument("--tier", choices=["L4", "L8", "L16", "L32"], default="L4")
    p.add_argument("--split", choices=["evolve", "gate", "test"], default="evolve",
                   help="level split (evolve=held-in / test=held-out / gate=fixed judge)")
    p.add_argument("--num-tasks", type=int, default=MAX_TASKS,
                   help="number of levels to roll out (0 = all levels in the cell's split)")
    p.add_argument("--episodes-per-level", type=int, default=4,
                   help="episodes per level (spec §5 seed scheme uses k = 0..N-1)")
    p.add_argument("--num-rounds", type=int, default=NUM_ROUNDS,
                   help="total rollout rounds (R0..R{N-1}); evolve runs after each non-last round")
    p.add_argument("--agent-model", default=AGENT_MODEL)
    p.add_argument("--agent-api-bases", default=AGENT_API_BASE,
                   help="comma-separated vLLM endpoints; rollout is sharded across them "
                        "(multi-GPU data parallelism)")
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--meta-model", default=DEFAULT_META_MODEL)
    p.add_argument("--meta-api-base", default=DEFAULT_META_API_BASE)
    p.add_argument("--evolve-cost", type=float, default=EVOLVE_COST_CAP_USD)
    p.add_argument("--evolve-steps", type=int, default=EVOLVE_MAX_STEPS)
    p.add_argument("--evolve-wall-clock", type=int, default=EVOLVE_WALL_CLOCK_S)
    p.add_argument("--regression-tolerance", type=float, default=REGRESSION_TOLERANCE)
    p.add_argument("--cost-weight", type=float, default=COST_WEIGHT)
    p.add_argument("--run-tag", default=None, help="output tag; defaults to <game>_<tier>_<split>_<timestamp>")
    p.add_argument("--base-config", default=None,
                   help="start from an external config YAML (e.g. an evolved config) "
                        "instead of the programmatic baseline — for held-out generalization probes")
    p.add_argument("--levels-file", default=None,
                   help="frozen jsonl of levels (make_gridgames_heldin_heldout.py output); "
                        "when set, --game/--tier/--split/--num-tasks are ignored for level selection")
    p.add_argument("--decision-backend", default="llm",
                   help="three-axis decision sidecar: 'llm' (off) or a jevlike name (e.g. 'kev')")
    p.add_argument("--decision-mode", default="prior", choices=("prior", "enforce"),
                   help="'prior' (soft, overrideable) or 'enforce' (hard lever bind + intent retrocheck)")
    p.add_argument("--decision-base-url", default="http://127.0.0.1:8090",
                   help="base URL of the decision backend (kev sidecar)")
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    game, tier, split = args.game, args.tier, args.split
    api_bases = [b.strip() for b in args.agent_api_bases.split(",") if b.strip()]

    # RAGen harness → switch the fixed parser + decoding for the whole run
    # (lazily read by gridgame_evolve, so this must be set before any rollout).
    if args.harness == "ragen":
        os.environ["EVOLVER_GRIDGAME_PARSER"] = "ragen"
        os.environ.setdefault("EVOLVER_GRIDGAME_MAX_TOKENS", "256")
        os.environ.setdefault("EVOLVER_GRIDGAME_ENABLE_THINKING", "0")
        logger.info("harness=ragen: parser=<answer> fullmatch, thinking off, max_tokens=%s",
                    os.environ["EVOLVER_GRIDGAME_MAX_TOKENS"])

    # The rollout cell: either a frozen --levels-file (three-axis held-in/held-out
    # datasets) or one (game × tier × split) slice of the manifest.
    if args.levels_file:
        levels = []
        for line in Path(args.levels_file).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                levels.append((d.get("_manifest_i", len(levels)), d))
        if not levels:
            raise SystemExit(f"no levels in --levels-file {args.levels_file}")
    else:
        all_levels = load_levels()
        levels = filter_levels(all_levels, game=game, tier=tier, split=split)
        if args.num_tasks:
            levels = levels[: args.num_tasks]
        if not levels:
            raise SystemExit(f"no levels for game={game} tier={tier} split={split}")

    # The move budget is uniform within a tier; used only to derive exit_reason.
    max_steps = max(lvl["budget"] for _, lvl in levels)

    run_tag = args.run_tag or f"{game}_{tier}_{split}_{time.strftime('%Y%m%d_%H%M%S')}"
    RUN_DIR = RUNS_DIR / "evolve_gridgames" / run_tag
    LEARNINGS_PATH = RUN_DIR / "learnings.md"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("run_tag=%s game=%s tier=%s split=%s levels=%d episodes/level=%d rounds=%d",
                run_tag, game, tier, split, len(levels), args.episodes_per_level, args.num_rounds)
    logger.info("agent=%s api_bases=%s concurrency=%d max_steps=%d",
                args.agent_model, ",".join(api_bases), args.concurrency, max_steps)

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
    else:
        current_config = make_gridgame_baseline_config(game, harness=args.harness)
    current_config = current_config.canonicalize()

    # ── Pre-flight: fail fast if enforce mode's decision backend is down ─────
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

    _timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
    _limits = httpx.Limits(max_connections=512, max_keepalive_connections=64)
    async with AsyncExitStack() as stack:
        clients = [
            await stack.enter_async_context(httpx.AsyncClient(timeout=_timeout, limits=_limits))
            for _ in api_bases
        ]
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

            # ── Rollout ─────────────────────────────────────────────────────
            spec = gridgame_spec_from_config(current_config)
            records = await run_gridgame_spec(
                clients,
                api_bases,
                args.agent_model,
                spec,
                levels,
                episodes_per_level=args.episodes_per_level,
                concurrency=args.concurrency,
            )
            left_out = sum(1 for r in records if r.get("left_out"))
            for rec in records:
                if not rec.get("left_out"):
                    write_task_trajectory(traj_dir, rec, max_steps)
                rec.pop("trajectory", None)  # keep memory light; trajectory is on disk
            gc.collect()
            all_rounds.append(records)

            passed = sum(1 for r in records if r["success"])
            avg_reward = sum(r["reward"] for r in records) / len(records) if records else 0.0
            round_cost = 0.0  # text loop has no cost accounting; cost_weight defaults to 0
            logger.info("[R%d] pass=%d/%d (%.3f)  mean_reward=%.4f  left_out=%d",
                        round_idx, passed, len(records),
                        (passed / len(records)) if records else 0.0, avg_reward, left_out)
            round_summaries.append({
                "round": round_idx, "config": config_label, "tasks": len(records),
                "passed": passed, "pass_rate": round((passed / len(records)), 3) if records else 0.0,
                "mean_reward": round(avg_reward, 4), "left_out": left_out,
            })

            # ── Gate ─────────────────────────────────────────────────────────
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

            # ── Journal back-fill (orchestrator contract, R1+) ──────────────
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

            # ── Evolve: produce next round's config ─────────────────────────
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
                        from .decision_prior import build_decision_priors

                        priors_path = evolve_dir / "_meta_scratch" / "decision_priors.md"
                        priors = await build_decision_priors(
                            benchmark="gridgames",
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
                                "(lever_argmax is None) — running unbound this round", round_idx)
                    except Exception as exc:  # noqa: BLE001
                        if args.decision_mode == "enforce":
                            logger.error(
                                "[R%d] enforce mode: decision backend %s unreachable (%s) — "
                                "cannot bind lever or retrocheck; aborting", round_idx, args.decision_backend, exc)
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

                # ── candidate retroactive-check (intent axis, enforce) ─────
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
                                round_idx, rejected, round_idx + 1)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[R%d] retrocheck skipped (non-fatal): %s", round_idx, exc)

            # Cooldown: the meta-agent bursts many requests; wait out rate-limit window.
            await asyncio.sleep(60)

    # ── Final report ─────────────────────────────────────────────────────────
    summary = {
        "game": game,
        "tier": tier,
        "split": split,
        "levels_file": args.levels_file,
        "run_tag": run_tag,
        "agent_model": args.agent_model,
        "num_levels": len(levels),
        "episodes_per_level": args.episodes_per_level,
        "rounds": round_summaries,
    }
    (RUN_DIR / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for s in round_summaries:
        logger.info("summary: R%(round)d %(config)s pass=%(passed)d/%(tasks)d rate=%(pass_rate)s "
                    "mean_reward=%(mean_reward)s left_out=%(left_out)s", s)
    logger.info("done → %s", RUN_DIR)


if __name__ == "__main__":
    asyncio.run(main())
