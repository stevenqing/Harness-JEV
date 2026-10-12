# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Unit tests for recipe.agent_evolver.layers (§2 / G1 of kev_ranker_step0c.md).

G1 = for each layer, construct a candidate with cross-layer changes, project it
onto that layer, and assert ``layers_of(parent, projected)`` is non-zero *only*
on the target layer.  Pure in-memory dicts — no rollout, no API, no model.
"""

from __future__ import annotations

import copy

import pytest

from recipe.agent_evolver.layers import LAYERS, layers_of, project


def _parent():
    return {
        "processors": [
            {
                "_target_": "harnessx.processors.TextEvolverSpecProcessor",
                "guidance": "old guidance text",
                "history_n": 2,
                "history_mode": "window",
                "tried_action_mask": False,
            },
            {
                "_target_": "harnessx.processors.SystemPromptProcessor",
                "system_builder": {"prompt": "old system prompt"},
            },
        ],
        "loop_breaker_k": 1,
        "tools": ["go_to", "take"],
    }


def _candidate():
    return {
        "processors": [
            {
                "_target_": "harnessx.processors.TextEvolverSpecProcessor",
                "guidance": "NEW guidance text (much longer than before)",
                "history_n": 8,
                "history_mode": "full",
                "tried_action_mask": True,
            },
            {
                "_target_": "harnessx.processors.SystemPromptProcessor",
                "system_builder": {"prompt": "NEW system prompt"},
            },
        ],
        "loop_breaker_k": 5,
        "tools": ["go_to", "take", "examine"],
    }


def test_candidate_touches_three_layers():
    sizes = layers_of(_parent(), _candidate())
    assert sizes["instruction"] > 0
    assert sizes["configuration"] > 0
    assert sizes["action"] > 0
    assert sizes["control"] == 0


@pytest.mark.parametrize("layer", ["instruction", "configuration", "action"])
def test_project_leaves_only_target_layer(layer):
    parent, cand = _parent(), _candidate()
    proj = project(parent, cand, layer)
    sizes = layers_of(parent, proj)
    assert sizes[layer] > 0, f"project({layer}) must keep that layer's change"
    for lyr in LAYERS:
        if lyr != layer:
            assert sizes[lyr] == 0, f"project({layer}) leaked {lyr}: {sizes}"


def test_project_instruction_reverts_scalars_and_tools():
    parent, cand = _parent(), _candidate()
    proj = project(parent, cand, "instruction")
    assert proj["processors"][0]["guidance"] == cand["processors"][0]["guidance"]
    assert proj["processors"][1]["system_builder"]["prompt"] == cand["processors"][1]["system_builder"]["prompt"]
    assert proj["processors"][0]["history_n"] == parent["processors"][0]["history_n"]
    assert proj["processors"][0]["history_mode"] == parent["processors"][0]["history_mode"]
    assert proj["processors"][0]["tried_action_mask"] == parent["processors"][0]["tried_action_mask"]
    assert proj["loop_breaker_k"] == parent["loop_breaker_k"]
    assert proj["tools"] == parent["tools"]  # "examine" (action layer) dropped


def test_project_control_layer():
    parent = {"processors": [{"_target_": "harnessx.processors.control.LoopBreaker", "k": 1, "label": "x"}]}
    cand = {"processors": [{"_target_": "harnessx.processors.control.LoopBreaker", "k": 5, "label": "y"}]}
    proj = project(parent, cand, "control")
    assert proj["processors"][0]["k"] == 5
    assert proj["processors"][0]["label"] == "y"
    sizes = layers_of(parent, proj)
    assert sizes["control"] > 0
    assert sizes["configuration"] == 0
    assert sizes["instruction"] == 0
    assert sizes["action"] == 0


def test_project_add_and_delete():
    parent = {"processors": [{"_target_": "x.Proc", "guidance": "old"}]}
    # candidate adds an instruction field, deletes nothing here; add a non-instruction add too
    cand = {
        "processors": [{"_target_": "x.Proc", "guidance": "new", "extra_scalar": 7}],
        "brand_new_field": "hello",
    }
    proj = project(parent, cand, "instruction")
    # "extra_scalar" and "brand_new_field" classify as configuration -> dropped
    assert "brand_new_field" not in proj
    assert "extra_scalar" not in proj["processors"][0]
    assert proj["processors"][0]["guidance"] == "new"
    sizes = layers_of(parent, proj)
    assert sizes["instruction"] > 0
    assert sizes["configuration"] == 0


def test_project_is_deterministic():
    parent, cand = _parent(), _candidate()
    a = project(parent, cand, "instruction")
    b = project(parent, cand, "instruction")
    assert a == b
    assert layers_of(parent, a) == layers_of(parent, b)


def test_project_unknown_layer_raises():
    with pytest.raises(ValueError):
        project(_parent(), _candidate(), "bogus")
