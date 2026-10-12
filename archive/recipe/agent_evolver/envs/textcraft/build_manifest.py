"""Build the fixed TextCraft task manifest (TEXTCRAFT_SPEC.md §2).

Reproducible across machines and PYTHONHASHSEED values: the task list is
sorted by ``(depth, goal)``, distractors are drawn with
``random.Random(hash_stable(task_id))`` where ``hash_stable`` is sha256's first
8 bytes (NOT Python's builtin ``hash``), and every set is sorted before it is
serialized. Run::

    .venv/bin/python -m recipe.agent_evolver.envs.textcraft.build_manifest
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Dict, List

from .vendor.crafting_tree import CraftingTree
from .vendor.utils import Recipe, item_id_to_str

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(HERE, "vendor")
MANIFEST_PATH = os.path.join(HERE, "manifest.jsonl")

_ORIG_LISTDIR = os.listdir


def _sorted_listdir(path="."):
    return sorted(_ORIG_LISTDIR(path))


def hash_stable(s: str) -> int:
    """sha256's first 8 bytes as an int — stable across runs and hash seeds."""
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def recipe_to_dict(recipe: Recipe) -> Dict:
    return {
        "output": recipe.output_item.item_tag.name,
        "count": recipe.output_item.count,
        "ingredients": [
            {"item": ic.item_tag.name, "count": ic.count} for ic in recipe.input_items
        ],
        "recipe_str": recipe.recipe_str,
    }


def dedupe_sorted(recipes: List[Recipe]) -> List[Recipe]:
    """Dedupe by recipe_str and sort — removes filesystem/hash-seed order."""
    by_str: Dict[str, Recipe] = {}
    for r in recipes:
        by_str.setdefault(r.recipe_str, r)
    return [by_str[k] for k in sorted(by_str)]


def gold_recipes(tree: CraftingTree, goal: str) -> List[Recipe]:
    return dedupe_sorted(tree.traverse_recipe_tree(goal))


def distractor_recipes(
    tree: CraftingTree, goal: str, rng: random.Random
) -> List[Recipe]:
    """Deterministic reproduction of CraftingTree.create_recipe_set's distractors.

    Same rule as upstream (for each input of a gold recipe, sample up to 10
    recipes that also use that input), but candidates are sorted before
    sampling and the draw uses the seeded rng — never ``random`` module state.
    """
    item_uses = tree.collect_item_uses()
    gold = gold_recipes(tree, goal)
    gold_strs = {r.recipe_str for r in gold}
    dist: Dict[str, Recipe] = {}
    for recipe in gold:
        for input_item in recipe.input_items:
            name = input_item.item_tag.name
            uses = dedupe_sorted(item_uses.get(name, []))
            n = min(len(uses), 10)
            for d in rng.sample(uses, n):
                if d.recipe_str not in gold_strs:
                    dist.setdefault(d.recipe_str, d)
    return [dist[k] for k in sorted(dist)]


def build() -> List[Dict]:
    # CraftingTree._load_recipes iterates os.listdir(recipes/) unsorted; the
    # recipe insertion order feeds cycle-removal (transitive_dependencies) and
    # therefore *which* recipes survive. Sort it so generation is independent
    # of the machine's directory-entry order (SPEC §2 "文件系统无关"), without
    # touching the vendored source.
    os.listdir = _sorted_listdir
    tree = CraftingTree(VENDOR_DIR)
    # all craftable items (depth >= 1), sorted by (depth, item_id == goal order)
    craftable = sorted(tree.item_recipes_min_depth(1), key=lambda x: (x[1], x[0]))

    rows: List[Dict] = []
    for i, (goal_id, depth) in enumerate(craftable):
        task_id = f"tc_{i:04d}"
        rng = random.Random(hash_stable(task_id))
        gold = gold_recipes(tree, goal_id)
        dist = distractor_recipes(tree, goal_id, rng)

        base: set = set()
        for r in gold:
            for ic in r.input_items:
                if not tree.is_craftable(ic.item_tag.name):
                    base.add(ic.item_tag.name)

        all_strs = sorted({r.recipe_str for r in gold} | {d.recipe_str for d in dist})
        rows.append(
            {
                "task_id": task_id,
                "goal": item_id_to_str(goal_id),
                "goal_id": goal_id,
                "depth": depth,
                "gold_recipes": [recipe_to_dict(r) for r in gold],
                "distractor_recipes": [recipe_to_dict(d) for d in dist],
                "commands_text": "\n".join(all_strs),
                "base_items": sorted(base),
            }
        )
    return rows


def main() -> None:
    rows = build()
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")

    by_depth: Dict[int, int] = {}
    for r in rows:
        by_depth[r["depth"]] = by_depth.get(r["depth"], 0) + 1
    print(f"manifest: {len(rows)} tasks -> {MANIFEST_PATH}")
    for d in sorted(by_depth):
        print(f"  depth {d}: {by_depth[d]}")


if __name__ == "__main__":
    main()
