# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Track B — B1: readout R0 (no evolution) on the 8-cell gate split.

A frozen readout agent (``semif`` = base-model first-token logit readout, or
``kev`` = the trained pointer-head sidecar) plays FrozenLake / Sokoban by
scoring the four moves each step and taking the argmax — no generation, no
decoding loop.  The question is the current board (bundle R0 renderer + prompt)
against four action options; the harness is the frozen ``r0.messages``.

    python -m recipe.agent_evolver.b1_readout --backend kev \
        --backend-base-url http://127.0.0.1:8090
    python -m recipe.agent_evolver.b1_readout --backend semif \
        --backend-base-url http://127.0.0.1:8200/v1 --model Qwen3.5-4B

Reports ``success − floors.gate.uniform`` per cell (2 games × 4 tiers) with a
95% percentile bootstrap over levels.  Episodes are 4 per level, written as one
JSON line each as soon as they end; restart skips finished episodes; infra
failures restart the episode from move 0 at most 3 times, then left out.

Model weights are never modified.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

from harnessx.decision import ChooseQ, get_decision_model

from . import r0
from .gridgame_evolve import filter_levels, load_levels

_BUNDLE_DIR = Path(os.environ.get(
    "EVOLVER_GRIDGAMES_DIR",
    "/mnt/llmshared-ssd-hd/shishuqing/gridgames",
)).resolve()
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))

import ta_env  # noqa: E402  (the bundle's LevelEnv, spec section 4)

EPISODES_PER_LEVEL = 4
MAX_EPISODE_ATTEMPTS = 4  # 1 initial + 3 restarts (spec section 10)
BOOTSTRAP_SEED = 20261007
BOOTSTRAP_DRAWS = 10_000

# The readout question is fixed in B1 (R0): the board + the four moves.  B2
# evolves the question wording / option grouping; this is the starting harness.
_CRITERION = "Choose the single move (up, down, left, or right) that best advances toward the goal."
_ACTION_OPTIONS = (("up", "move up"), ("down", "move down"),
                   ("left", "move left"), ("right", "move right"))
_QUESTION = ChooseQ(_CRITERION, _ACTION_OPTIONS)

# HTTP statuses that are transient infrastructure problems (restart the episode).
_INFRA_STATUS = {429, 500, 502, 503, 504}
_FATAL_STATUS = {400, 404, 422}


class InfraError(Exception):
    """Transient infrastructure failure — restart the episode from move 0."""


class FatalError(Exception):
    """Permanent configuration error — abort the run."""


def readout_state(obs: dict, history: list) -> str:
    """The evidence the readout scores: the R0 system prompt + board + last moves."""
    msgs = r0.messages(obs, history)
    return "\n\n".join(m["content"] for m in msgs)


async def run_episode(backend, level: dict, line_i: int, k: int) -> dict:
    """One readout episode: score the four moves each step, take the argmax."""
    env = ta_env.LevelEnv(level)
    history: list = []
    moves: list = []
    while not env.done:
        obs = env.observation()
        m = len(moves)
        try:
            decisions = await backend.decide(
                state=readout_state(obs, history), questions=[_QUESTION]
            )
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError,
                httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout,
                httpx.WriteError) as exc:
            raise InfraError(str(exc)) from exc
        dec = decisions[0]
        action = dec.argmax
        probs = {name: round(dec.probabilities.get(name, 0.0), 6) for name, _ in _ACTION_OPTIONS}
        env.step(action)
        moves.append({
            "m": m,
            "action": action,
            "probs": probs,
            "confidence": round(dec.confidence, 6),
            "last_result": env.last_result,
        })
        history.append((action, env.last_result))
    rep = env.report()
    return {
        "backend": backend.name,
        "level_id": level["id"],
        "level_line": line_i,
        "episode": k,
        "episode_seed": 1000 * line_i + k,
        "game": level["game"],
        "tier": level["tier"],
        "split": level["split"],
        "left_out": False,
        "attempts": 1,
        "success": bool(rep["success"]),
        "report": rep,
        "moves": moves,
    }


async def run_episode_with_retry(backend, level, line_i, k) -> dict:
    for attempt in range(1, MAX_EPISODE_ATTEMPTS + 1):
        try:
            rec = await run_episode(backend, level, line_i, k)
            rec["attempts"] = attempt
            return rec
        except InfraError:
            if attempt >= MAX_EPISODE_ATTEMPTS:
                return {
                    "backend": backend.name, "level_id": level["id"], "level_line": line_i,
                    "episode": k, "episode_seed": 1000 * line_i + k,
                    "game": level["game"], "tier": level["tier"], "split": level["split"],
                    "left_out": True, "attempts": attempt, "success": False,
                }
            await asyncio.sleep(5.0 * attempt)
    raise AssertionError("unreachable")


def _floors(bundle: Path) -> dict:
    tiers = json.loads((bundle / "tiers.json").read_text())
    return tiers["floors"]["gate"]


def _bootstrap_mean_ci(samples: list[float], draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float]:
    """95% percentile bootstrap over the (per-level) sample values."""
    import random

    rng = random.Random(BOOTSTRAP_SEED)
    n = len(samples)
    if n == 0:
        return 0.0, 0.0
    means = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(sum(samples[i] for i in idx) / n)
    means.sort()
    lo = means[int(0.025 * draws)]
    hi = means[int(0.975 * draws)]
    return lo, hi


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="kev", help="readout backend name (kev | semif)")
    p.add_argument("--backend-base-url", default="http://127.0.0.1:8090",
                   help="backend sidecar base URL (kev sidecar or vLLM /v1)")
    p.add_argument("--model", default=None, help="served model name (semif: Qwen3.5-4B; kev: kev-latest)")
    p.add_argument("--games", default="frozenlake,sokoban")
    p.add_argument("--tiers", default="L4,L8,L16,L32")
    p.add_argument("--episodes-per-level", type=int, default=EPISODES_PER_LEVEL)
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--out-dir", default="runs/agent_evolver/gridgames/b1_readout")
    p.add_argument("--retry-leftout", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="smoke cap: run at most N episodes (0 = all)")
    p.add_argument("--bundle", type=Path, default=_BUNDLE_DIR)
    a = p.parse_args(argv)

    games = [g for g in a.games.split(",") if g]
    tiers = [t for t in a.tiers.split(",") if t]

    model = a.model or ("Qwen3.5-4B" if a.backend == "semif" else "kev-latest")
    backend = get_decision_model(a.backend, base_url=a.backend_base_url, model=model)

    out_path = Path(a.out_dir) / f"{a.backend}" / "episodes.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("left_out") and not a.retry_leftout:
                done.add((rec["level_id"], rec["episode"]))
            elif not rec.get("left_out"):
                done.add((rec["level_id"], rec["episode"]))

    levels = load_levels(a.bundle)
    gate = []
    for line_i, lvl in levels:
        if lvl["split"] != "gate" or lvl["game"] not in games or lvl["tier"] not in tiers:
            continue
        gate.append((line_i, lvl))

    work = []
    for line_i, lvl in gate:
        for k in range(a.episodes_per_level):
            if (lvl["id"], k) not in done:
                work.append((line_i, lvl, k))
    if a.limit:
        work = work[: a.limit]

    print(f"[b1_readout] backend={a.backend} url={a.backend_base_url} model={model}", flush=True)
    print(f"[b1_readout] gate levels={len(gate)} episodes={len(done) + len(work)} to_run={len(work)}", flush=True)
    if not work:
        print("[b1_readout] nothing to do (all episodes already recorded)", flush=True)
        return _report(out_path, a.bundle, games, tiers)

    lock = asyncio.Lock()
    sem = asyncio.Semaphore(a.concurrency)
    counters = {"finished": 0, "left_out": 0}
    t0 = time.time()

    async def worker(backend_ctx, item):
        line_i, lvl, k = item
        async with sem:
            rec = await run_episode_with_retry(backend_ctx, lvl, line_i, k)
        async with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if rec.get("left_out"):
                counters["left_out"] += 1
            else:
                counters["finished"] += 1
            n = counters["finished"] + counters["left_out"]
            if n % 100 == 0:
                print(f"[b1_readout] {counters['finished']} finished / "
                      f"{counters['left_out']} left_out / elapsed {time.time() - t0:.0f}s", flush=True)

    async def run_all():
        try:
            async with backend:
                await asyncio.gather(*(worker(backend, item) for item in work))
        except FatalError as e:
            print(f"[b1_readout] FATAL: {e}", flush=True)
            return 2
        return 0

    rc = asyncio.run(run_all())
    print(f"[b1_readout] DONE finished={counters['finished']} left_out={counters['left_out']} "
          f"elapsed {time.time() - t0:.0f}s", flush=True)
    _report(out_path, a.bundle, games, tiers)
    return rc


def _report(out_path: Path, bundle: Path, games: list[str], tiers: list[str]) -> int:
    floors = _floors(bundle)
    recs = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not recs:
        print("[b1_readout] no records yet", flush=True)
        return 0

    # per-cell: group per-level episode success → bootstrap over levels
    cells: dict[tuple[str, str], dict] = {}
    for g in games:
        for t in tiers:
            cells[(g, t)] = {"successes": [], "n_ep": 0, "levels": {}}

    for r in recs:
        if r.get("left_out"):
            continue
        cell = cells[(r["game"], r["tier"])]
        cell["levels"].setdefault(r["level_id"], []).append(1.0 if r["success"] else 0.0)

    print("\n# B1 readout — success − floors.gate.uniform (per cell)", flush=True)
    print("| game | tier | success | floor.uniform | Δ | 95% CI |", flush=True)
    print("|---|---|---|---|---|---|", flush=True)
    for g in games:
        for t in tiers:
            cell = cells[(g, t)]
            level_rates = [sum(v) / len(v) for v in cell["levels"].values()]
            mean = sum(level_rates) / len(level_rates) if level_rates else 0.0
            lo, hi = _bootstrap_mean_ci(level_rates)
            floor = floors[g][t]["uniform"]
            print(f"| {g} | {t} | {mean:.3f} | {floor:.4f} | {mean - floor:+.3f} | "
                  f"[{lo:.3f}, {hi:.3f}] |", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
