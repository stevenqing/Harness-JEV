# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Spec-driven rollout for FrozenLake / Sokoban (Phase C).

This is the gridgame analogue of ``agents.run_alfworld`` / ``agents.run_webshop``:
it runs the R0 loop of spec section 5 with a *decoded* ``GridgameSpec`` instead
of the frozen R0 prompt.  The only things the spec changes are the system prompt
(+ guidance) and the history window; the board renderer (``r0.render_board``)
and the action parser (``r0.parse_action``) stay fixed.

    obs = env.observation()
    msgs = messages_from_spec(obs, history, spec)   # 2 messages: system + user
    reply = <model>(msgs)                            # one model call per move
    action = r0.parse_action(reply)                  # fixed parser, unchanged
    env.step(action)

The model call reuses ``baseline_gridgames.py``'s raw-HTTP ``_chat`` (spec §5
seed scheme, infra-retry, thinking off + max_tokens 32) so the baseline and the
evolved rollout differ **only** in the harness text, never in decoding.

The adapter imports ``ta_env`` from the bundle directory, which stays outside
every path the meta-agent can read (spec section 6.1).  The adapter is recipe
code, not harness code (spec section 6.2).

Model weights are never modified.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from . import r0
from .gridgame_spec import GridgameSpec

# The bundle directory lives OUTSIDE the repo tree (spec section 6.1).  Only
# this adapter imports from it; the meta-agent can never read it.
_BUNDLE_DIR = Path(os.environ.get(
    "EVOLVER_GRIDGAMES_DIR",
    "/mnt/llmshared-ssd-hd/shishuqing/gridgames",
)).resolve()
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))

import ta_env  # noqa: E402  (the bundle's LevelEnv, spec section 4)

# ── decoding (spec section 5, byte-identical to baseline_gridgames.py) ────────
TEMPERATURE = 0.4
TOP_P = 0.8
TOP_K = 20
MAX_TOKENS = 4096
ENABLE_THINKING = True

# Per-run harness/decoding overrides (lazily read so run_gridgames.py can switch
# to the RAGen-style harness — thinking OFF + short budget + <answer> parser —
# without restarting the process or touching module-level constants).
def _max_tokens() -> int:
    return int(os.environ.get("EVOLVER_GRIDGAME_MAX_TOKENS", str(MAX_TOKENS)))


def _enable_thinking() -> bool:
    v = os.environ.get("EVOLVER_GRIDGAME_ENABLE_THINKING", "1" if ENABLE_THINKING else "0")
    return v == "1"


def _parse_action():
    """Return the fixed action parser: RAGen's ``<answer>`` fullmatch when the
    ``EVOLVER_GRIDGAME_PARSER=ragen`` override is set, else R0's bracket parser."""
    if os.environ.get("EVOLVER_GRIDGAME_PARSER") == "ragen":
        from .ragen_harness import parse_action as _ragen_parse_action
        return _ragen_parse_action
    return r0.parse_action

EPISODES_PER_LEVEL = 4
MAX_EPISODE_ATTEMPTS = 4  # 1 initial + 3 restarts (spec section 10)

# HTTP statuses that are a transient infrastructure problem (restart the episode).
_INFRA_STATUS = {429, 500, 502, 503, 504}
# Statuses that mean our request is malformed — abort the whole run.
_FATAL_STATUS = {400, 404, 422}


class InfraError(Exception):
    """Transient infrastructure failure — restart the episode from move 0."""


class FatalError(Exception):
    """Permanent configuration error — abort the run."""


# ── level manifest ────────────────────────────────────────────────────────────

def load_levels(bundle_dir: Path | str | None = None) -> list[tuple[int, dict]]:
    """Return ``(i, level)`` for every manifest line, ``i`` the 0-based line."""
    p = Path(bundle_dir) if bundle_dir else _BUNDLE_DIR
    return [
        (i, json.loads(line))
        for i, line in enumerate((p / "levels.jsonl").read_text().splitlines())
        if line.strip()
    ]


def filter_levels(
    levels: list[tuple[int, dict]],
    *,
    game: str,
    tier: str | None = None,
    split: str | None = None,
) -> list[tuple[int, dict]]:
    """Keep the ``(line_i, level)`` entries matching game / tier / split."""
    out = []
    for line_i, lvl in levels:
        if lvl["game"] != game:
            continue
        if tier is not None and lvl["tier"] != tier:
            continue
        if split is not None and lvl["split"] != split:
            continue
        out.append((line_i, lvl))
    return out


# ── spec-driven messages ──────────────────────────────────────────────────────

def _fill_placeholders(system: str, rows: int, budget: int) -> str:
    """Substitute the two known R0 placeholders; leave everything else intact."""
    return system.replace("{size}", str(rows)).replace("{budget}", str(budget))


def _render_user(obs: dict, history: list, history_window: int) -> str:
    board = r0.render_board(obs)
    lines = ["Board:", board, "", f"Moves left: {obs['moves_left']}"]
    if history:
        lines.append("Last moves:")
        for action, result in history[-history_window:]:
            a = action if action is not None else "(unreadable)"
            lines.append(f"  {a} -> {result}")
    return "\n".join(lines)


def messages_from_spec(obs: dict, history: list, spec: GridgameSpec) -> list[dict]:
    """Two messages (system + user) driven by the evolvable spec.

    ``history`` = list of ``(action, result)``.  The system prompt is the spec's
    ``system_prompt`` with ``{size}``/``{budget}`` substituted and ``guidance``
    appended (kept inside the single system message to preserve the R0 two-
    message contract).  The board is drawn by the fixed ``r0.render_board``.
    """
    system = _fill_placeholders(spec.system_prompt, obs["rows"], obs["budget"])
    if spec.guidance:
        system = f"{system}\n\n{spec.guidance}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _render_user(obs, history, spec.history_window)},
    ]


# ── model call ───────────────────────────────────────────────────────────────

async def _chat(client: httpx.AsyncClient, api_base: str, model: str,
                msgs: list[dict], seed: int) -> str:
    """One model call for one move.  Returns the reply text, or raises
    ``InfraError`` on a transient failure / ``FatalError`` on a bad request."""
    body = {
        "model": model,
        "messages": msgs,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_tokens": _max_tokens(),
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": _enable_thinking()},
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


# ── one episode ──────────────────────────────────────────────────────────────

async def run_episode(client: httpx.AsyncClient, api_base: str, model: str,
                      spec: GridgameSpec, level: dict, line_i: int, k: int) -> dict:
    """One R0 episode with the decoded spec.  Returns the rollout-shape record
    (``task_id`` = ``<level id>_e<k>`` so the four episodes of a level do not
    collide in the trajectories dir).  Raises ``InfraError`` to restart from
    move 0."""
    episode_seed = 1000 * line_i + k
    env = ta_env.LevelEnv(level)
    history: list = []
    traj: list = []
    parse_action = _parse_action()  # fixed parser (bracket or RAGen <answer>)
    while not env.done:
        obs = env.observation()
        msgs = messages_from_spec(obs, history, spec)
        m = len(traj)  # move index within the episode, 0-based
        request_seed = 100 * episode_seed + m
        raw = await _chat(client, api_base, model, msgs, request_seed)
        action = parse_action(raw)  # str | None (parser fixed)
        env.step(action)
        traj.append({
            "step": m + 1,
            "obs": msgs[-1]["content"],  # the readable board the model saw
            "raw": raw,
            "action": action,
            "invalid": action is None,
            "last_result": env.last_result,
        })
        history.append((action, env.last_result))
    rep = env.report()
    final_obs = messages_from_spec(env.observation(), history, spec)[-1]["content"]
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
        "goal": f"{level['game']} {level['tier']} level {level['id']}",
        "trajectory": traj,
        "final_obs": final_obs,
        "report": rep,
    }


def _left_out_record(level: dict, k: int, attempts: int) -> dict:
    """Placeholder record for an episode that never completed (infra only)."""
    return {
        "task_id": f"{level['id']}_e{k}",
        "level_id": level["id"],
        "game": level["game"],
        "tier": level["tier"],
        "split": level["split"],
        "order": level["order"],
        "opt_steps": level["opt_steps"],
        "success": False,
        "reward": 0.0,
        "steps": 0,
        "goal": f"{level['game']} {level['tier']} level {level['id']}",
        "trajectory": [],
        "final_obs": "",
        "left_out": True,
        "attempts": attempts,
    }


async def run_episode_with_retry(client, api_base, model, spec, level, line_i, k) -> dict:
    for attempt in range(1, MAX_EPISODE_ATTEMPTS + 1):
        try:
            return await run_episode(client, api_base, model, spec, level, line_i, k)
        except InfraError:
            if attempt >= MAX_EPISODE_ATTEMPTS:
                return _left_out_record(level, k, attempt)
            await asyncio.sleep(5.0 * attempt)
    raise AssertionError("unreachable")


# ── rollout over a level list ────────────────────────────────────────────────

async def run_gridgame_spec(
    clients: list[httpx.AsyncClient],
    api_bases: list[str],
    model: str,
    spec: GridgameSpec,
    levels: list[tuple[int, dict]],
    *,
    episodes_per_level: int = EPISODES_PER_LEVEL,
    concurrency: int = 48,
) -> list[dict]:
    """Roll out ``spec`` over ``levels``, sharding levels round-robin across the
    ``clients``/``api_bases`` endpoints (parallel lists, one vLLM server each —
    multi-GPU data parallelism).  Returns one record per episode.  Raises
    ``FatalError`` on a bad request to abort the whole round.

    Seeds are keyed by ``line_i`` (manifest index), so sharding levels across
    endpoints is deterministic and replay-safe.
    """
    n = len(clients)
    assert n == len(api_bases) and n >= 1
    shards: list[list[tuple[int, dict]]] = [[] for _ in range(n)]
    for i, item in enumerate(levels):
        shards[i % n].append(item)

    async def run_shard(shard, client, api_base) -> list[dict]:
        work = [(line_i, lvl, k) for line_i, lvl in shard for k in range(episodes_per_level)]
        results: list[dict | None] = [None] * len(work)
        sem = asyncio.Semaphore(concurrency)

        async def run_one(idx: int, item) -> None:
            line_i, lvl, k = item
            async with sem:
                results[idx] = await run_episode_with_retry(client, api_base, model, spec, lvl, line_i, k)

        await asyncio.gather(*(run_one(i, it) for i, it in enumerate(work)))
        return [r for r in results if r is not None]

    shard_results = await asyncio.gather(
        *(run_shard(shards[i], clients[i], api_bases[i]) for i in range(n))
    )
    return [r for results in shard_results for r in results]
