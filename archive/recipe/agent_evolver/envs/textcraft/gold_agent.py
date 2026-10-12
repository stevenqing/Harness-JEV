"""Gold agent for TextCraft (TEXTCRAFT_SPEC.md §9 G1).

Replays the gold recipe tree topologically: ``get`` every base item, then
``craft`` intermediates in dependency order, resolving tag ingredients to a
specific craftable item and computing the exact invocation counts the
environment needs (one recipe output per ``craft``). Must reach 100% on the
held-in + held-out 128 tasks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .vendor.utils import item_id_to_str


def build_plan(tree, row: Dict) -> List[str]:
    """Return the concrete command list that crafts ``row``'s goal."""
    goal_id = row["goal_id"]
    gold = row["gold_recipes"]

    by_output: Dict[str, List[Dict]] = {}
    for r in gold:
        by_output.setdefault(r["output"], []).append(r)

    def recipe_weight(r: Dict) -> int:
        return max(
            (tree.get_min_depth(ing["item"]) for ing in r["ingredients"]), default=0
        )

    chosen: Dict[str, Dict] = {}
    tag_map: Dict[str, str] = {}

    def resolve_tag(tag: str) -> str:
        if tag not in tag_map:
            cands = sorted(c for c in tree.get_items_with_tags(tag) if c in by_output)
            if not cands:
                raise RuntimeError(f"tag {tag} has no craftable item in the recipe set")
            tag_map[tag] = cands[0]
        return tag_map[tag]

    def resolved_ings(r: Dict):
        out = []
        for ing in r["ingredients"]:
            name = ing["item"]
            if tree.is_tag(name):
                name = resolve_tag(name)
            out.append((name, ing["count"]))
        return out

    def choose(item_id: str) -> None:
        if item_id in chosen:
            return
        recipes = by_output.get(item_id)
        if not recipes:
            return
        best = min(recipes, key=lambda r: (recipe_weight(r), r["recipe_str"]))
        chosen[item_id] = best
        for name, _c in resolved_ings(best):
            if name in by_output:
                choose(name)

    choose(goal_id)
    if goal_id not in chosen:
        raise RuntimeError(f"goal {goal_id} has no gold recipe")

    # post-order (children first) topological sort
    visited = set()
    order: List[str] = []

    def dfs(item: str) -> None:
        if item in visited:
            return
        visited.add(item)
        recipe = chosen.get(item)
        if recipe is not None:
            for name, _c in resolved_ings(recipe):
                dfs(name)
        order.append(item)

    dfs(goal_id)

    # top-down demand propagation (goal first)
    need = defaultdict(int)
    need[goal_id] = 1
    invocations: Dict[str, int] = {}
    for item in reversed(order):
        recipe = chosen.get(item)
        if recipe is None:
            continue
        out_count = recipe["count"]
        inv = -(-need[item] // out_count)  # ceil
        invocations[item] = inv
        for name, c in resolved_ings(recipe):
            need[name] += inv * c

    commands: List[str] = []
    for n in sorted(n for n in need if n not in by_output):
        if need[n] > 0:
            commands.append(f"get {need[n]} {item_id_to_str(n)}")
    for item in order:
        recipe = chosen.get(item)
        if recipe is None:
            continue
        ing_parts = ", ".join(
            f"{c} {item_id_to_str(name)}" for name, c in resolved_ings(recipe)
        )
        cmd = (
            f"craft {recipe['count']} {item_id_to_str(recipe['output'])} "
            f"using {ing_parts}"
        )
        for _ in range(invocations[item]):
            commands.append(cmd)
    return commands
