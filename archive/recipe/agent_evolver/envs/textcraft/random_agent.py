"""Random floor agent for TextCraft (TEXTCRAFT_SPEC.md §9 G1).

Uniformly samples from the admissible action list each step, for at most
MAX_STEPS steps. Used to report the success floor.
"""

from __future__ import annotations

import random
from typing import Dict

from .env import MAX_STEPS, TextCraftEnvWrapper


class RandomAgent:
    def __init__(self, wrapper: TextCraftEnvWrapper):
        self.wrapper = wrapper

    def run(self, task_id: str, rng: random.Random) -> bool:
        self.wrapper.reset(task_id)
        for _ in range(MAX_STEPS):
            adms = self.wrapper.admissible or ["inventory"]
            action = rng.choice(adms)
            _obs, _reward, done, _info = self.wrapper.step(action)
            if done:
                return True
            if self.wrapper.should_early_stop:
                break
        return False
