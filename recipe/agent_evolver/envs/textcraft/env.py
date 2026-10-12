"""In-process TextCraft wrapper (TEXTCRAFT_SPEC.md §3).

Same interface as the ALFWorld / WebShop wrappers: ``reset(task_id) ->
(obs, info)`` and ``step(action) -> (obs, reward, done, info)``. A single
shared CraftingTree + a single vendored TextCraftEnv are reused across
episodes; ``reset`` loads the goal + commands text from the fixed manifest so
no random task generation (or global random.seed) is involved.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
from typing import Dict, List, Optional

from .vendor.crafting_tree import CraftingTree
from .vendor.environment import TextCraftEnv
from .vendor.utils import item_id_to_str

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(HERE, "vendor")
MANIFEST_PATH = os.path.join(HERE, "manifest.jsonl")

_ID_RE = re.compile(r"minecraft:([a-z0-9_]+)")


def _normalize_obs(s: str) -> str:
    return _ID_RE.sub(lambda m: m.group(1).replace("_", " "), s)


def load_manifest() -> Dict[str, Dict]:
    manifest: Dict[str, Dict] = {}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            manifest[row["task_id"]] = row
    return manifest


class TextCraftEnvWrapper:
    def __init__(
        self,
        manifest: Optional[Dict[str, Dict]] = None,
        tree: Optional[CraftingTree] = None,
    ):
        self.manifest = manifest if manifest is not None else load_manifest()
        self.tree = tree if tree is not None else CraftingTree(VENDOR_DIR)
        self.env = TextCraftEnv(self.tree, commands="", goal="")
        self.task_id: Optional[str] = None
        self.current_row: Optional[Dict] = None
        self.step_idx = 0
        self.admissible: List[str] = []

    def _build_admissible(self, row: Dict) -> List[str]:
        actions: List[str] = ["inventory"]
        recipe_set = row["gold_recipes"] + row["distractor_recipes"]
        craft_cmds: List[str] = []
        seen = set()
        for r in recipe_set:
            s = r["recipe_str"]
            if s not in seen:
                seen.add(s)
                craft_cmds.append(s)
        outputs = {r["output"] for r in recipe_set}
        get_cmds = set()
        for r in recipe_set:
            for ing in r["ingredients"]:
                name = ing["item"]
                count = ing["count"]
                if self.tree.is_tag(name):
                    for specific in self.tree.get_items_with_tags(name):
                        if specific in outputs and not self.tree.is_craftable(specific):
                            get_cmds.add(f"get {count} {item_id_to_str(specific)}")
                elif not self.tree.is_craftable(name):
                    get_cmds.add(f"get {count} {item_id_to_str(name)}")
        actions.extend(sorted(craft_cmds))
        actions.extend(sorted(get_cmds))
        return actions

    def reset(self, task_id: str) -> tuple[str, Dict]:
        row = self.manifest[task_id]
        self.task_id = task_id
        self.current_row = row
        self.step_idx = 0
        obs, _ = self.env.reset(commands=row["commands_text"], goal=row["goal_id"])
        self.admissible = self._build_admissible(row)
        info = {
            "inventory": {},
            "env_message": obs,
            "in_admissible": True,
            "step_idx": self.step_idx,
        }
        return _normalize_obs(obs), info

    def step(self, action: str) -> tuple[str, float, bool, Dict]:
        self.step_idx += 1
        if action == "":
            obs = "Invalid action format."
            done = False
            reward = 0.0
            info = {
                "inventory": dict(self.env.inventory),
                "env_message": obs,
                "in_admissible": False,
                "step_idx": self.step_idx,
            }
            return obs, reward, done, info

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            obs, reward, terminated, truncated, _info = self.env.step(action)
        done = bool(terminated or truncated)
        in_admissible = action in self.admissible
        info = {
            "inventory": dict(self.env.inventory),
            "env_message": obs,
            "in_admissible": in_admissible,
            "step_idx": self.step_idx,
        }
        return _normalize_obs(obs), float(reward), done, info
