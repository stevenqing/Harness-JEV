# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Gridgame (FrozenLake / Sokoban) environment adapter and R0 runner.

This is the analogue of ``agents.run_alfworld`` / ``agents.run_webshop`` for the
two bundle games.  It wraps the bundle's ``LevelEnv`` (spec section 4 episode
contract) with the R0 loop of spec section 5:

    obs = env.observation()
    msgs = r0.messages(obs, history)      # exactly 2 messages: system + user
    reply = <model>(msgs)                 # one model call per move
    action = r0.parse_action(reply)       # fixed parser, unchanged every round
    env.step(action)

``r0`` is the copied ``r0.py`` (byte-identical to the bundle's; spec section 5).
The adapter imports ``ta_env`` from the bundle directory, which stays outside
every path the meta-agent can read (spec section 6.1).  The adapter is recipe
code, not harness code (spec section 6.2), so importing ``ta_env`` here is legal
— the meta-agent's *harness* may not.

For gate G1 the model call is one of the bundle's stubs (``StubProvider``); for
phase B it is the real agent model served by vLLM through a ``LiteLLMProvider``.

Model weights are never modified.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from harnessx.core.events import Message

from . import r0

# The bundle directory lives OUTSIDE the repo tree (spec section 6.1).  Only
# this adapter imports from it; the meta-agent can never read it.
_BUNDLE_DIR = Path(os.environ.get(
    "EVOLVER_GRIDGAMES_DIR",
    "/mnt/llmshared-ssd-hd/shishuqing/gridgames",
)).resolve()
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))

import stubs  # noqa: E402  (the bundle's four stub models, spec section 5)
import ta_env  # noqa: E402  (the bundle's LevelEnv, spec section 4)


def load_levels(bundle_dir: Path | str | None = None) -> list[dict]:
    p = Path(bundle_dir) if bundle_dir else _BUNDLE_DIR
    return [json.loads(line) for line in (p / "levels.jsonl").read_text().splitlines() if line.strip()]


def load_tiers(bundle_dir: Path | str | None = None) -> dict:
    p = Path(bundle_dir) if bundle_dir else _BUNDLE_DIR
    return json.loads((p / "tiers.json").read_text())


def gate_levels(levels: list[dict]) -> list[dict]:
    return [lvl for lvl in levels if lvl["split"] == "gate"]


class StubProvider:
    """Adapt a bundle stub (callable ``messages -> reply``) to the runner's
    model-call interface.

    The stub receives the complete, unfiltered list of chat messages of the
    request (spec section 5).  ``calls`` mirrors the stub's own counter.
    """

    def __init__(self, stub):
        self.stub = stub

    @property
    def calls(self) -> int:
        return self.stub.calls

    async def reply(self, msgs: list[dict]) -> str:
        return self.stub(msgs)


async def _model_reply(model, msgs: list[dict]) -> str:
    """One model call for one move.  ``msgs`` is ``r0.messages``' two dicts."""
    if isinstance(model, StubProvider):
        return await model.reply(msgs)
    message_objs = [Message(role=m["role"], content=m["content"]) for m in msgs]
    resp = await model.complete(message_objs, [])
    return resp.content


async def run_gridgame(model, levels: list[dict]) -> list[dict]:
    """Run one R0 episode per level.  Returns one record per level in the
    recipe's rollout shape (``task_id`` = level ``id``), carrying the full
    ``LevelEnv.report()`` metrics under ``report``.

    R0 has no code after the parser, so the parser output *is* the move passed
    to ``step`` (spec section 6.2).
    """
    records: list[dict] = []
    for level in levels:
        env = ta_env.LevelEnv(level)
        history: list = []
        traj: list = []
        while not env.done:
            obs = env.observation()
            msgs = r0.messages(obs, history)
            reply = await _model_reply(model, msgs)
            action = r0.parse_action(reply)
            env.step(action)
            traj.append({
                "step": len(traj) + 1,
                "messages": msgs,
                "obs": obs,
                "raw": reply,
                "parser_output": action,
                "action": action,
                "last_result": env.last_result,
                "invalid": action is None,
            })
            history.append((action, env.last_result))
        rep = env.report()
        records.append({
            "task_id": level["id"],
            "level_id": level["id"],
            "game": level["game"],
            "tier": level["tier"],
            "split": level["split"],
            "order": level["order"],
            "opt_steps": level["opt_steps"],
            "success": bool(rep["success"]),
            "reward": float(rep["progress_end"]),
            "steps": rep["moves"],
            "goal": level["id"],
            "trajectory": traj,
            "final_obs": env.observation(),
            "report": rep,
        })
    return records
