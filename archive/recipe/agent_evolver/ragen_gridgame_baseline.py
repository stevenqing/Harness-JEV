# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""RAGen-style baseline for FrozenLake / Sokoban (R0, no evolution).

Applies RAGen's rollout *protocol* to our gridgames ``LevelEnv`` so we can
measure the starting pass rate under RAGen's harness vs our bracket-based
harness (``r0``).  The differences from ``gridgame_evolve`` are exactly the
RAGen recipe:

* prompt   : system = game rules + symbols (bracket hint stripped);
             user   = board + moves left + last moves + RAGen's format directive
             ("Always output: <think> ... </think> <answer> up|down|left|right </answer>
              with no extra text. Strictly follow this format.").
* decoding : ``enable_thinking=False`` (prompt-level ``<think>`` tag, no Qwen
             native thinking) + a short ``max_tokens`` budget.
* parser   : extract the ``<answer>`` payload, ``fullmatch`` one direction word,
             else ``None`` (LevelEnv treats ``None`` as a no-op / parse_fail —
             the analogue of RAGen's "invalid action").

The env, levels, seeds and sharding are byte-identical to ``gridgame_evolve``,
so the numbers are directly comparable to the three-axis ``llm`` arm.
Model weights are never modified.
"""
from __future__ import annotations

import asyncio
import argparse
import json
import os
import re
import sys
from pathlib import Path

import httpx

from . import r0

# The bundle directory lives OUTSIDE the repo tree (spec section 6.1).
_BUNDLE_DIR = Path(os.environ.get(
    "EVOLVER_GRIDGAMES_DIR",
    "/mnt/llmshared-ssd-hd/shishuqing/gridgames",
)).resolve()
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))

import ta_env  # noqa: E402  (the bundle's LevelEnv, spec section 4)

# ── decoding: RAGen-style (thinking OFF, short response budget) ────────────────
TEMPERATURE = float(os.environ.get("RAGEN_TEMPERATURE", "0.4"))
TOP_P = float(os.environ.get("RAGEN_TOP_P", "0.8"))
TOP_K = int(os.environ.get("RAGEN_TOP_K", "20"))
MAX_TOKENS = int(os.environ.get("RAGEN_MAX_TOKENS", "256"))
ENABLE_THINKING = False  # prompt-level <think> tag, NOT Qwen native thinking

_INFRA_STATUS = {429, 500, 502, 503, 504}
_FATAL_STATUS = {400, 404, 422}

ACTIONS = ["up", "down", "left", "right"]

# ── RAGen harness text ─────────────────────────────────────────────────────────

FORMAT_DIRECTIVE = (
    "Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> "
    "with no extra text. Strictly follow this format. "
    "Your answer must be exactly one of: up, down, left, right."
)


def system_message(game: str, budget: int, size: int) -> str:
    if game == "frozenlake":
        return (
            "You are playing FrozenLake on a {size}x{size} grid of ice. "
            "Each turn move up, down, left or right to walk the player to the goal.\n"
            "Symbols: P = player, F = ice (safe), H = hole (fall in and lose), G = goal (win).\n"
            "Reach G without stepping on H. You have {budget} moves.\n"
            "Your move must be exactly one of: up, down, left, right."
        ).format(size=size, budget=budget)
    return (
        "You are playing Sokoban. Push every box onto a target to win.\n"
        "Symbols: # = wall, _ = floor, O = target, P = player, S = player standing on a target, "
        "B = box, X = box on a target.\n"
        "Move up, down, left or right. A move into a wall, off the board, or pushing a box into a "
        "wall or another box does nothing.\n"
        "You have {budget} moves. Your move must be exactly one of: up, down, left, right."
    ).format(budget=budget)


def messages(obs: dict, history: list) -> list[dict]:
    """Two messages (system + user).  ``history`` = list of ``(action, result)``."""
    system = system_message(obs["game"], obs["budget"], obs["rows"])
    board = r0.render_board(obs)

    lines = ["Board:", board, "", f"Moves left: {obs['moves_left']}"]
    if history:
        lines.append("Last moves:")
        for action, result in history[-3:]:
            a = action if action is not None else "(invalid)"
            lines.append(f"  {a} -> {result}")
    user = "\n".join(lines) + "\n\n" + FORMAT_DIRECTIVE
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── RAGen-style action parser ─────────────────────────────────────────────────

_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_DIRECTION = re.compile(r"^\s*(up|down|left|right)\s*$", re.IGNORECASE)


def parse_action(reply: str) -> str | None:
    """Extract the ``<answer>`` payload and ``fullmatch`` it against one
    direction word (RAGen ``extract_action`` semantics).  ``None`` = invalid
    (LevelEnv no-op)."""
    m = _ANSWER.search(reply)
    payload = m.group(1).strip() if m else reply.strip()
    mm = _DIRECTION.fullmatch(payload)
    return mm.group(1).lower() if mm else None


# ── model call (mirrors gridgame_evolve._chat, ragen decoding) ────────────────

class InfraError(Exception):
    pass


class FatalError(Exception):
    pass


async def _chat(client: httpx.AsyncClient, api_base: str, model: str,
                msgs: list[dict], seed: int) -> str:
    body = {
        "model": model,
        "messages": msgs,
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
        last = RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        await asyncio.sleep(2.0 * (attempt + 1))
    raise InfraError(f"model call failed after 3 attempts: {last}")


# ── one episode ────────────────────────────────────────────────────────────────

async def run_episode(client: httpx.AsyncClient, api_base: str, model: str,
                      level: dict, line_i: int, k: int) -> dict:
    episode_seed = 1000 * line_i + k
    env = ta_env.LevelEnv(level)
    history: list = []
    traj: list = []
    while not env.done:
        obs = env.observation()
        msgs = messages(obs, history)
        m = len(traj)
        request_seed = 100 * episode_seed + m
        raw = await _chat(client, api_base, model, msgs, request_seed)
        action = parse_action(raw)  # str | None
        env.step(action)
        traj.append({
            "step": m + 1,
            "obs": msgs[-1]["content"],
            "raw": raw,
            "action": action,
            "invalid": action is None,
            "last_result": env.last_result,
        })
        history.append((action, env.last_result))
    rep = env.report()
    return {
        "task_id": f"{level['id']}_e{k}",
        "level_id": level["id"],
        "game": level["game"],
        "tier": level["tier"],
        "split": level["split"],
        "order": level["order"],
        "opt_steps": level["opt_steps"],
        "success": bool(rep["success"]),
        "reward": float(rep["progress_end"]),
        "steps": rep["moves"],
        "parse_fail": rep["parse_fail"],
        "blocked": rep["blocked_moves"],
        "goal": f"{level['game']} {level['tier']} level {level['id']}",
        "trajectory": traj,
        "report": rep,
    }


# ── rollout over a level list ────────────────────────────────────────────────

async def run_spec(clients, api_bases, model, levels, episodes_per_level, concurrency):
    n = len(clients)
    assert n == len(api_bases) and n >= 1
    shards = [[] for _ in range(n)]
    for i, item in enumerate(levels):
        shards[i % n].append(item)

    async def run_shard(shard, client, api_base):
        work = [(line_i, lvl, k) for line_i, lvl in shard for k in range(episodes_per_level)]
        results = [None] * len(work)
        sem = asyncio.Semaphore(concurrency)

        async def run_one(idx, item):
            line_i, lvl, k = item
            async with sem:
                results[idx] = await run_episode(client, api_base, model, lvl, line_i, k)

        await asyncio.gather(*(run_one(i, it) for i, it in enumerate(work)))
        return [r for r in results if r is not None]

    shard_results = await asyncio.gather(
        *(run_shard(shards[i], clients[i], api_bases[i]) for i in range(n))
    )
    return [r for results in shard_results for r in results]


def _load_levels_file(path: str) -> list[tuple[int, dict]]:
    levels = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            levels.append((d.get("_manifest_i", len(levels)), d))
    return levels


def _summarize(records: list[dict], tag: str) -> dict:
    n = len(records)
    n_success = sum(1 for r in records if r["success"])
    n_left_out = sum(1 for r in records if r.get("left_out"))
    reward = sum(r["reward"] for r in records) / n if n else 0.0
    moves = sum(r["steps"] for r in records)
    parse_fail = sum(r["parse_fail"] for r in records)
    blocked = sum(r["blocked"] for r in records)
    return {
        "tag": tag,
        "n_episodes": n,
        "success": n_success,
        "pass_rate": round(n_success / n, 4) if n else 0.0,
        "mean_reward": round(reward, 4),
        "mean_moves": round(moves / n, 2) if n else 0.0,
        "total_moves": moves,
        "parse_fail_moves": parse_fail,
        "parse_fail_fraction": round(parse_fail / moves, 4) if moves else 0.0,
        "blocked_moves": blocked,
        "left_out": n_left_out,
    }


async def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels-file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--agent-api-bases", required=True,
                    help="comma-separated vLLM /v1 endpoints")
    ap.add_argument("--model", default="Qwen3.5-4B")
    ap.add_argument("--episodes-per-level", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--out", default=None, help="json summary output path")
    args = ap.parse_args()

    levels = _load_levels_file(args.levels_file)
    api_bases = [b.strip() for b in args.agent_api_bases.split(",") if b.strip()]
    timeout = httpx.Timeout(600.0, connect=60.0)
    limits = httpx.Limits(max_connections=args.concurrency + 4,
                          max_keepalive_connections=args.concurrency + 4)
    clients = [httpx.AsyncClient(timeout=timeout, limits=limits) for _ in api_bases]

    try:
        records = await run_spec(clients, api_bases, args.model, levels,
                                 args.episodes_per_level, args.concurrency)
    finally:
        for c in clients:
            await c.aclose()

    summ = _summarize(records, args.tag)
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summ, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
