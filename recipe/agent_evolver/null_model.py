# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Null-model runners (visibility spec S4).

The null policy is the uniform-random policy: uniform over the four gridgame
moves, or uniform over the current admissible command in ALFWorld/WebShop.
Every evolved harness is also played by this policy on the *same split* so the
report can show the null column next to the model's column.

gridgames runs in-process through the bundle's ``ta_env`` (no model call).
ALFWorld/WebShop reuse the env servers but never call the model — each step
picks a uniform random admissible command.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import httpx

from . import agents

_BUNDLE_DIR = Path(os.environ.get(
    "EVOLVER_GRIDGAMES_DIR", "/mnt/llmshared-ssd-hd/shishuqing/gridgames")).resolve()
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))
import ta_env  # noqa: E402  (bundle LevelEnv)
import gridgames as gg  # noqa: E402  (bundle transition rules: gg.ACTIONS)


# ── gridgames: uniform random move, in-process ───────────────────────────────

def run_null_gridgames(
    levels: list[tuple[int, dict]],
    *,
    episodes_per_level: int = 4,
    seed: int = 1234,
) -> list[dict]:
    """Play uniform-random-move null policy over ``(line_i, level)`` entries.

    Returns records shaped like ``gridgame_evolve.run_episode`` (``level_id``,
    ``success``, ``reward``=progress_end, ``report``) so the report code can
    reuse the same bootstrap/report path.  No model call is made.
    """
    rng = random.Random(seed)
    records: list[dict] = []
    for line_i, level in levels:
        for k in range(episodes_per_level):
            env = ta_env.LevelEnv(level)
            while not env.done:
                env.step(rng.choice(gg.ACTIONS))
            rep = env.report()
            records.append({
                "task_id": f"{level['id']}_e{k}",
                "level_id": level["id"],
                "game": level["game"],
                "tier": level["tier"],
                "split": level["split"],
                "opt_steps": level["opt_steps"],
                "success": bool(rep["success"]),
                "reward": float(rep["progress_end"]),
                "steps": rep["moves"],
                "report": rep,
            })
    return records


# ── ALFWorld / WebShop: uniform random admissible command ────────────────────

async def _null_alfworld_one(client: httpx.AsyncClient, gi: int, seed: int, max_steps: int) -> dict:
    rng = random.Random(1000 * seed + gi)
    r = (await client.post("/reset", json={"game_idx": gi, "seed": seed})).json()
    obs = r["observation"]
    adm = r.get("admissible_commands", [])
    done = False
    won = False
    steps = 0
    while not done and steps < max_steps and adm:
        action = rng.choice(adm)
        r = (await client.post("/step", json={"action": action})).json()
        obs = r["observation"]
        adm = r.get("admissible_commands", [])
        won = bool(r.get("won"))
        done = bool(r.get("done"))
        steps += 1
    return {"task_id": gi, "success": won, "reward": 1.0 if won else 0.0, "steps": steps}


async def run_null_alfworld(
    game_idxs: list[int],
    *,
    max_steps: int = 50,
    seed: int = 1234,
    env_url: str | None = None,
) -> list[dict]:
    """Uniform-random-admissible-command null policy over ALFWorld game_idxs."""
    url = env_url or agents.ALFWORLD_ENV_URL
    records: list[dict] = []
    async with httpx.AsyncClient(base_url=url, timeout=180.0) as client:
        for gi in game_idxs:
            records.append(await _null_alfworld_one(client, gi, seed, max_steps))
    return records


async def _null_webshop_one(client: httpx.AsyncClient, tid: int, seed: int, max_steps: int) -> dict:
    rng = random.Random(2000 * seed + tid)
    c = (await client.post("/create", json={"task_id": tid, "seed": seed})).json()
    env_id = c["env_id"]
    avail = c["available_actions"]
    done = False
    steps = 0
    reward = 0.0
    while not done and steps < max_steps:
        action_list = agents._ws_action_list(avail)
        if not action_list:
            break
        action = rng.choice(action_list)
        r = (await client.post("/step", json={"env_id": env_id, "action": action})).json()
        avail = r.get("available_actions", {})
        reward = float(r["reward"])
        done = bool(r["done"])
        steps += 1
    await client.post("/destroy", json={"env_id": env_id})
    return {"task_id": tid, "success": reward >= 0.999, "reward": reward, "steps": steps}


async def run_null_webshop(
    task_ids: list[int],
    *,
    max_steps: int = 15,
    seed: int = 0,
    env_url: str | None = None,
) -> list[dict]:
    """Uniform-random-admissible-command null policy over WebShop task_ids."""
    url = env_url or agents.WEBSHOP_ENV_URL
    records: list[dict] = []
    async with httpx.AsyncClient(base_url=url, timeout=180.0) as client:
        for tid in task_ids:
            records.append(await _null_webshop_one(client, tid, seed, max_steps))
    return records
