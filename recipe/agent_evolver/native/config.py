# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Text-protocol ``HarnessConfig`` encoding for ALFWorld / WebShop.

The agent is the weak Qwen3.5-4B model driven by the Skill1 ``<think>/<action>``
ReAct loop (``agents.run_alfworld`` / ``agents.run_webshop``), **not** native
tool-calling.  The ``HarnessConfig`` is a pure *carrier* of the evolvable spec:

1. ``SystemPromptProcessor`` + ``StaticSystemPromptBuilder`` → ``system_prompt``.
2. ``TextEvolverSpecProcessor`` → ``guidance`` (the ``{skills}`` slot) + the
   flattened loop-processor params (loop-breaker / history-format /
   tried-action-mask).

The rollout reads the spec back via :func:`spec_from_config`; the config itself
is only ever *edited* by ``MetaAgent.evolve()`` and *replayed* by the validator
(a no-op, non-crashing pipeline).
"""
from __future__ import annotations

from pathlib import Path

from harnessx.core.harness import HarnessConfig

from recipe.agent_evolver.processors import PROCESSOR_DEFAULTS
from recipe.agent_evolver.spec import QWEN_SYSTEM, HarnessSpec

_NATIVE_DIR = Path(__file__).resolve().parent
_PROMPT_PATH = _NATIVE_DIR / "prompt.py"
_TEXT_SPEC_PATH = _NATIVE_DIR / "text_spec.py"

_TEXT_SPEC_FIELDS = (
    "guidance",
    "loop_breaker_k",
    "history_mode",
    "history_n",
    "history_obs_clip",
    "tried_action_mask",
)


def _system_prompt_processor(prompt: str) -> dict:
    return {
        "_target_": "harnessx.processors.context.system_prompt.SystemPromptProcessor",
        "system_builder": {
            "_target_": f"file://{_PROMPT_PATH}::StaticSystemPromptBuilder",
            "prompt": prompt,
        },
    }


def _text_spec_processor(guidance: str, processors: dict) -> dict:
    """Flatten ``HarnessSpec.processors`` into scalar constructor kwargs."""
    merged: dict[str, dict] = {}
    for name, defaults in PROCESSOR_DEFAULTS.items():
        d = dict(defaults)
        d.update(processors.get(name) or {})
        merged[name] = d
    return {
        "_target_": f"file://{_TEXT_SPEC_PATH}::TextEvolverSpecProcessor",
        "guidance": guidance,
        "loop_breaker_k": int(merged["loop_breaker"]["k"]),
        "history_mode": merged["history_format"]["mode"],
        "history_n": int(merged["history_format"]["n"]),
        "history_obs_clip": int(merged["history_format"]["obs_clip"]),
        "tried_action_mask": bool(merged["tried_action_mask"]["enabled"]),
    }


def _config_for_spec(spec: HarnessSpec) -> HarnessConfig:
    return HarnessConfig(
        processors=[
            _system_prompt_processor(spec.system_prompt),
            _text_spec_processor(spec.guidance, spec.processors),
        ],
    )


def make_alfworld_baseline_config() -> HarnessConfig:
    """Baseline = Skill1/verl-agent base: Qwen system prompt, empty guidance."""
    return _config_for_spec(HarnessSpec(system_prompt=QWEN_SYSTEM, guidance=""))


def make_webshop_baseline_config() -> HarnessConfig:
    return _config_for_spec(HarnessSpec(system_prompt=QWEN_SYSTEM, guidance=""))


# ── Config → HarnessSpec ─────────────────────────────────────────────────────


def _find_system_prompt(cfg: HarnessConfig) -> str:
    for proc in cfg.processors:
        if isinstance(proc, dict) and "system_builder" in proc:
            builder = proc["system_builder"]
            if isinstance(builder, dict) and "prompt" in builder:
                return str(builder["prompt"])
    return QWEN_SYSTEM


def _find_text_spec(cfg: HarnessConfig) -> dict:
    defaults: dict = {
        "guidance": "",
        "loop_breaker_k": int(PROCESSOR_DEFAULTS["loop_breaker"]["k"]),
        "history_mode": PROCESSOR_DEFAULTS["history_format"]["mode"],
        "history_n": int(PROCESSOR_DEFAULTS["history_format"]["n"]),
        "history_obs_clip": int(PROCESSOR_DEFAULTS["history_format"]["obs_clip"]),
        "tried_action_mask": bool(PROCESSOR_DEFAULTS["tried_action_mask"]["enabled"]),
    }
    for proc in cfg.processors:
        target = proc.get("_target_", "") if isinstance(proc, dict) else ""
        if target.endswith("::TextEvolverSpecProcessor"):
            for k in _TEXT_SPEC_FIELDS:
                if k in proc:
                    defaults[k] = proc[k]
            break
    return defaults


def spec_from_config(cfg: HarnessConfig) -> HarnessSpec:
    """Decode the evolvable text-loop spec back out of a (possibly evolved) config."""
    prompt = _find_system_prompt(cfg)
    ts = _find_text_spec(cfg)
    processors = {
        "loop_breaker": {"k": int(ts["loop_breaker_k"])},
        "history_format": {
            "mode": ts["history_mode"],
            "n": int(ts["history_n"]),
            "obs_clip": int(ts["history_obs_clip"]),
        },
        "tried_action_mask": {"enabled": bool(ts["tried_action_mask"])},
    }
    return HarnessSpec(system_prompt=prompt, guidance=str(ts["guidance"]), processors=processors)
