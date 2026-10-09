# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Gridgame (FrozenLake / Sokoban) ``HarnessConfig`` encoding.

The gridgame loop (R0, spec section 5) is driven by our own rollout code, **not**
the native tool-calling runloop.  The ``HarnessConfig`` is a pure *carrier* of
the evolvable spec: a single ``GridgameSpecProcessor`` holds ``system_prompt`` +
``guidance`` + ``history_window`` (see ``gridgame_spec.py``).  The rollout reads
the spec back via :func:`gridgame_spec_from_config`; the config itself is only
ever *edited* by ``MetaAgent.evolve()`` and *replayed* by the validator (a no-op,
non-crashing pipeline).

Unlike the ALF/WS encoding, the system prompt is **not** split out into a
``SystemPromptProcessor`` + ``StaticSystemPromptBuilder`` pair — one processor
carries the whole gridgame spec, which keeps the diff the meta-agent sees to a
single ``_target_`` block.
"""
from __future__ import annotations

from pathlib import Path

from harnessx.core.harness import HarnessConfig

from recipe.agent_evolver.gridgame_spec import GridgameSpec, baseline_system_prompt
from recipe.agent_evolver.ragen_harness import baseline_system_prompt as ragen_baseline_system_prompt

_NATIVE_DIR = Path(__file__).resolve().parent
_GRIDGAME_SPEC_PROC_PATH = _NATIVE_DIR / "gridgame_spec_processor.py"

_GRIDGAME_SPEC_FIELDS = ("system_prompt", "guidance", "history_window")


def _gridgame_spec_processor(spec: GridgameSpec) -> dict:
    return {
        "_target_": f"file://{_GRIDGAME_SPEC_PROC_PATH}::GridgameSpecProcessor",
        "system_prompt": spec.system_prompt,
        "guidance": spec.guidance,
        "history_window": int(spec.history_window),
    }


def make_gridgame_baseline_config(game: str, harness: str = "bracket") -> HarnessConfig:
    """Baseline = the R0 prompt for ``game``, empty guidance, history window 3.

    ``harness`` selects the starting prompt: ``"bracket"`` (R0's "[up]") or
    ``"ragen"`` (RAGen's ``<think>/<answer>`` protocol)."""
    prompt = ragen_baseline_system_prompt(game) if harness == "ragen" else baseline_system_prompt(game)
    spec = GridgameSpec(system_prompt=prompt)
    return HarnessConfig(processors=[_gridgame_spec_processor(spec)])


# ── Config → GridgameSpec ─────────────────────────────────────────────────────


def gridgame_spec_from_config(cfg: HarnessConfig) -> GridgameSpec:
    """Decode the evolvable gridgame spec back out of a (possibly evolved) config."""
    defaults: dict = {"system_prompt": "", "guidance": "", "history_window": 3}
    for proc in cfg.processors:
        target = proc.get("_target_", "") if isinstance(proc, dict) else ""
        if target.endswith("::GridgameSpecProcessor"):
            for k in _GRIDGAME_SPEC_FIELDS:
                if k in proc:
                    defaults[k] = proc[k]
            break
    return GridgameSpec(
        system_prompt=str(defaults["system_prompt"]),
        guidance=str(defaults["guidance"]),
        history_window=int(defaults["history_window"]),
    )
