# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Phase B R0 baseline for FrozenLake / Sokoban (spec section 7 Phase B).

Runs the R0 loop — ``r0.messages`` / ``r0.parse_action`` / ``ta_env.LevelEnv`` —
over the gate split of all 8 cells for the three Qwen3.5 models served by vLLM,
using the seed scheme of spec section 5::

    episode_seed = 1000 * i + k        (i = 0-based line in levels.jsonl, k = episode 0..3)
    request seed = 100 * episode_seed + m     (m = move 0.. within the episode)

Each episode is one JSON line, written as soon as it ends; on restart finished
episodes are skipped.  An episode that fails for an *infrastructure* reason
(connection / timeout / HTTP 5xx / 429) is restarted from its first move with
the same seeds, at most 3 restarts, then left out (spec section 10).  An
unreadable reply is *not* an error — the parser returns ``None`` and ``step``
counts a ``parse_fail`` move.

    python -m recipe.agent_evolver.baseline_gridgames \
        --models Qwen3.5-4B,Qwen3.5-2B,Qwen3.5-0.8B \
        --api-bases http://127.0.0.1:8200/v1,http://127.0.0.1:8201/v1,http://127.0.0.1:8202/v1 \
        --concurrency 48

Model weights are never modified.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from pathlib import Path

import httpx

from . import r0

# The R0 module used for messages/parsing.  ``--r0-module`` swaps this to an
# experimental variant (e.g. ``r0_textarena``) without touching the frozen spec R0.
R0 = r0

# The bundle directory lives OUTSIDE the repo tree (spec section 6.1).
_BUNDLE_DIR = Path(os.environ.get(
    "EVOLVER_GRIDGAMES_DIR",
    "/mnt/llmshared-ssd-hd/shishuqing/gridgames",
)).resolve()
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))

import ta_env  # noqa: E402  (the bundle's LevelEnv, spec section 4)

# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_MODELS = ["Qwen3.5-4B", "Qwen3.5-2B", "Qwen3.5-0.8B"]
DEFAULT_API_BASES = {
    "Qwen3.5-4B": "http://127.0.0.1:8200/v1",
    "Qwen3.5-2B": "http://127.0.0.1:8201/v1",
    "Qwen3.5-0.8B": "http://127.0.0.1:8202/v1",
}

# Decoding (spec section 5).  Thinking is off (vLLM ``chat_template_kwargs``).
# ``runs/agent_evolver/official/`` does not exist on this box, so sampling is the
# recipe's own 4B generative baseline (spec.py: temp 0.4 / top_p 0.8), which is a
# single, non-zero-temperature configuration, so the section-5 fallback (0.7 /
# top_k 20) is not triggered; top_k 20 is still pinned for reproducibility.
TEMPERATURE = 0.4
TOP_P = 0.8
TOP_K = 20
MAX_TOKENS = 32
ENABLE_THINKING = False  # spec §5 hardcodes thinking off; flip to test decoding

EPISODES_PER_LEVEL = 4
MAX_EPISODE_ATTEMPTS = 4  # 1 initial + 3 restarts (spec section 10)

# HTTP statuses that are a transient infrastructure problem (restart the episode).
_INFRA_STATUS = {429, 500, 502, 503, 504}
# Statuses that mean our request is malformed (e.g. a serving flag rejected) —
# not transient, abort the whole run rather than mark 6144 episodes "left out".
_FATAL_STATUS = {400, 404, 422}


class InfraError(Exception):
    """Transient infrastructure failure — restart the episode from move 0."""


class FatalError(Exception):
    """Permanent configuration error — abort the run."""


def load_levels() -> list[tuple[int, dict]]:
    """Return ``(i, level)`` for every manifest line, ``i`` the 0-based line."""
    return [
        (i, json.loads(line))
        for i, line in enumerate((_BUNDLE_DIR / "levels.jsonl").read_text().splitlines())
        if line.strip()
    ]


def gate_levels(levels: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
    return [(i, lvl) for i, lvl in levels if lvl["split"] == "gate"]


# ── model call ───────────────────────────────────────────────────────────────

async def _chat(client: httpx.AsyncClient, api_base: str, model: str,
                msgs: list[dict], seed: int) -> str:
    """One model call for one move.  Returns the reply text, or raises
    ``InfraError`` on a transient failure / ``FatalError`` on a bad request."""
    body = {
        "model": model,
        "messages": msgs,  # the two r0 messages: system + user
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
    }
    last: BaseException | None = None
    for attempt in range(3):
        try:
            r = await client.post(f"{api_base}/chat/completions", json=body)
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError,
                httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout,
                httpx.WriteError) as exc:
            last = exc
            await asyncio.sleep(2.0 * (attempt + 1))
            continue
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"] or ""
        if r.status_code in _INFRA_STATUS:
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
            await asyncio.sleep(2.0 * (attempt + 1))
            continue
        if r.status_code in _FATAL_STATUS:
            raise FatalError(f"HTTP {r.status_code}: {r.text[:400]}")
        # Anything else is unexpected but transient-ish; treat as infra.
        last = RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        await asyncio.sleep(2.0 * (attempt + 1))
    raise InfraError(f"model call failed after 3 attempts: {last}")


# ── one episode ──────────────────────────────────────────────────────────────

async def run_episode(client: httpx.AsyncClient, api_base: str, model: str,
                      level: dict, line_i: int, k: int) -> dict:
    """One R0 episode.  Returns the JSON record.  Raises ``InfraError`` to
    trigger a restart from move 0."""
    episode_seed = 1000 * line_i + k
    env = ta_env.LevelEnv(level)
    history: list = []
    moves: list = []
    while not env.done:
        obs = env.observation()
        msgs = R0.messages(obs, history)
        m = len(moves)  # move index within the episode, 0-based
        request_seed = 100 * episode_seed + m
        raw = await _chat(client, api_base, model, msgs, request_seed)
        action = R0.parse_action(raw)  # str | None
        env.step(action)
        moves.append({
            "m": m,
            "request_seed": request_seed,
            "raw": raw,
            "parser_output": action,
            "last_result": env.last_result,
        })
        history.append((action, env.last_result))
    rep = env.report()
    return {
        "model": model,
        "level_id": level["id"],
        "level_line": line_i,
        "episode": k,
        "episode_seed": episode_seed,
        "game": level["game"],
        "tier": level["tier"],
        "order": level["order"],
        "left_out": False,
        "attempts": 1,
        "report": rep,
        "moves": moves,
    }


async def run_episode_with_retry(client, api_base, model, level, line_i, k) -> dict:
    for attempt in range(1, MAX_EPISODE_ATTEMPTS + 1):
        try:
            rec = await run_episode(client, api_base, model, level, line_i, k)
            rec["attempts"] = attempt
            return rec
        except InfraError:
            if attempt >= MAX_EPISODE_ATTEMPTS:
                return {
                    "model": model, "level_id": level["id"], "level_line": line_i,
                    "episode": k, "episode_seed": 1000 * line_i + k,
                    "game": level["game"], "tier": level["tier"], "order": level["order"],
                    "left_out": True, "attempts": attempt, "reason": "infra",
                }
            await asyncio.sleep(5.0 * attempt)
    raise AssertionError("unreachable")


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    global R0, MAX_TOKENS, ENABLE_THINKING
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--api-bases", default=None,
                   help="comma list aligned with --models; default from DEFAULT_API_BASES")
    p.add_argument("--episodes-per-level", type=int, default=EPISODES_PER_LEVEL)
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--out-dir", default="runs/agent_evolver/gridgames/baseline")
    p.add_argument("--retry-leftout", action="store_true",
                   help="re-attempt episodes previously marked left_out")
    p.add_argument("--limit", type=int, default=0,
                   help="smoke-test cap: run at most this many episodes (0 = all)")
    p.add_argument("--r0-module", default="r0",
                   help="messages/parser module inside this package (default r0; "
                        "use r0_textarena for the TextArena-aligned experiment)")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                   help="max_tokens per move (spec §5 pins 32; enlarge to test decoding)")
    p.add_argument("--enable-thinking", action="store_true",
                   help="turn model thinking on (spec §5 pins off)")
    p.add_argument("--tiers", default=None,
                   help="comma list of tiers to restrict to (default: all)")
    a = p.parse_args(argv)

    R0 = importlib.import_module(f".{a.r0_module}", package=__package__)
    MAX_TOKENS = a.max_tokens
    ENABLE_THINKING = a.enable_thinking

    models = [m for m in a.models.split(",") if m]
    if a.api_bases:
        bases = [b for b in a.api_bases.split(",") if b]
    else:
        bases = [DEFAULT_API_BASES[m] for m in models]
    if len(bases) != len(models):
        raise SystemExit("--models and --api-bases must have the same length")

    out_path = Path(a.out_dir) / "episodes.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip episodes that already have a line.
    done: set = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("left_out") and not a.retry_leftout:
                done.add((rec["model"], rec["level_id"], rec["episode"]))
            elif not rec.get("left_out"):
                done.add((rec["model"], rec["level_id"], rec["episode"]))

    levels = load_levels()
    gl = gate_levels(levels)
    if a.tiers:
        want = {t for t in a.tiers.split(",") if t}
        gl = [(i, lvl) for i, lvl in gl if lvl["tier"] in want]
    work = []
    for model, api_base in zip(models, bases):
        for line_i, lvl in gl:
            for k in range(a.episodes_per_level):
                if (model, lvl["id"], k) not in done:
                    work.append((model, api_base, line_i, lvl, k))

    if a.limit:
        work = work[: a.limit]

    print(f"[baseline] models={models} bases={bases}", flush=True)
    print(f"[baseline] gate levels={len(gl)}  total episodes={len(done) + len(work)}  "
          f"to run={len(work)}  concurrency={a.concurrency}", flush=True)
    if not work:
        print("[baseline] nothing to do (all episodes already recorded)", flush=True)
        return 0

    lock = asyncio.Lock()
    sem = asyncio.Semaphore(a.concurrency)
    counters = {"finished": 0, "left_out": 0}
    t0 = time.time()

    async def worker(client, item):
        model, api_base, line_i, lvl, k = item
        async with sem:
            rec = await run_episode_with_retry(client, api_base, model, lvl, line_i, k)
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
                print(f"[baseline] {counters['finished']} finished / "
                      f"{counters['left_out']} left_out / elapsed {time.time() - t0:.0f}s",
                      flush=True)

    async def run_all():
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=64),
        ) as client:
            try:
                await asyncio.gather(*(worker(client, item) for item in work))
            except FatalError as e:
                print(f"[baseline] FATAL (aborting run): {e}", flush=True)
                return 2
        return 0

    rc = asyncio.run(run_all())

    print(f"[baseline] DONE  finished={counters['finished']}  left_out={counters['left_out']}  "
          f"elapsed {time.time() - t0:.0f}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
