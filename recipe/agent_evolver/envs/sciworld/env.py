"""Thin ScienceWorld HTTP client wrapper (SCIWORLD_SPEC.md §4).

Mirrors the ``reset(episode_id) -> (obs, info)`` / ``step(action_text) ->
(obs, reward, done, info)`` contract of the in-process TextCraft wrapper, but
the environment itself lives behind ``benchmarks/sciworld/env_server.py`` (a
py3.8 process pool).  This wrapper is *synchronous* (``httpx.Client``); the
async runner calls it through ``asyncio.to_thread``.

Client-side responsibilities (everything the thin server does NOT do):

  * §4.1  reset observation = task description + initial look.
  * §4.3  ambiguity menu auto-answered ``0`` (recorded as ``auto_disambiguated``,
         not counted as an agent step).
  * §4.4  score = ``max(final_score_raw, 0) / 100``; ``max_score`` tracked;
         ``success`` is ``final_score_raw == 100``.
  * §4.2  ``possible_actions`` / ``possible_objects`` surfaced for the prompt;
         ``in_valid`` (exact member of the pre-step valid combinations) recorded,
         never shown to the agent.
"""

from __future__ import annotations

import os
from typing import Dict, List

import httpx

SCIWORLD_ENV_URL = os.environ.get("EVOLVER_SCIWORLD_ENV_URL", "http://127.0.0.1:18091")

AMBIGUITY_PREFIX = "Ambiguous request"
MAX_AUTO_ANSWERS = 3


def compose_reset_obs(task_desc: str, look: str) -> str:
    return f"{task_desc}\n\n{look}"


def is_ambiguous(obs: str) -> bool:
    return str(obs).strip().startswith(AMBIGUITY_PREFIX)


def clip_score(score_raw: int) -> float:
    return max(score_raw, 0) / 100.0


class ScienceWorldClient:
    def __init__(self, base_url: str = SCIWORLD_ENV_URL, timeout: float = 180.0):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self.env_id: int | None = None
        self.task_desc = ""
        self.task_name: str | None = None
        self.variation: int | None = None
        self.possible_actions: List[str] = []
        self.possible_objects: List[str] = []
        self.score_raw = 0
        self.score = 0.0
        self.max_score = 0

    def reset(self, episode_id: str) -> tuple[str, Dict]:
        r = self._client.post("/create", json={"episode_id": episode_id}).json()
        self.env_id = r["env_id"]
        self.task_desc = r["task_desc"]
        self.task_name = r["task_name"]
        self.variation = r["variation"]
        self.possible_actions = r["possible_actions"]
        self.possible_objects = r["possible_objects"]
        self.score_raw = 0
        self.score = 0.0
        self.max_score = 0
        obs = compose_reset_obs(self.task_desc, r["observation"])
        info = {
            "score_raw": 0,
            "score": 0.0,
            "in_valid": True,
            "auto_disambiguated": False,
            "moves": r["moves"],
        }
        return obs, info

    def step(self, action: str) -> tuple[str, float, bool, Dict]:
        r = self._client.post("/step", json={"env_id": self.env_id, "action": action}).json()
        in_valid = bool(r["in_valid"])
        obs = r["observation"]
        auto_disambiguated = False
        n = 0
        while is_ambiguous(obs) and n < MAX_AUTO_ANSWERS:
            r = self._client.post("/step", json={"env_id": self.env_id, "action": "0"}).json()
            obs = r["observation"]
            auto_disambiguated = True
            n += 1
        self._update(r, obs)
        info = {
            "score_raw": self.score_raw,
            "score": self.score,
            "in_valid": in_valid,
            "auto_disambiguated": auto_disambiguated,
            "moves": r["moves"],
        }
        return obs, self.score, bool(r["done"]), info

    def _update(self, r: Dict, obs: str) -> None:
        self.score_raw = int(r["score"])
        self.score = clip_score(self.score_raw)
        self.max_score = max(self.max_score, max(self.score_raw, 0))
        self.possible_actions = r["possible_actions"]
        self.possible_objects = r["possible_objects"]

    def goal_progress(self) -> str:
        return self._client.post("/goal_progress", json={"env_id": self.env_id}).json()["goal_progress"]

    def destroy(self) -> None:
        if self.env_id is not None:
            try:
                self._client.post("/destroy", json={"env_id": self.env_id})
            except Exception:  # noqa: BLE001 — teardown best-effort
                pass
            self.env_id = None

    def close(self) -> None:
        self._client.close()
