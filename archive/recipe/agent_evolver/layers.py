"""Layer definition for harness config diffs — the four lever layers.

Migrated from ``data/kev_ranker_s0b/layers.py`` (§5.3) per ``specs/kev_ranker_step0c.md`` §2.
Provides the deterministic mapping ``layers_of`` (unchanged from the 0b version)
plus ``project``, which clips a candidate config down to a single layer's changes.

Layers (see ``layers.md`` for the field→layer table the user reviews):

  1. action        — field path contains ``tools`` / ``tool_registry`` / ``actions``
                     (the action-space / tool surface).
  2. control       — changed field lives inside a ``processors[]`` entry whose
                     ``_target_`` references ``processors.control.`` (a mechanical
                     post-processing hook).
  3. instruction   — field path contains ``prompt`` or ``guidance`` (free-text
                     instruction surfaces: system prompt + per-turn guidance).
  4. configuration — every other changed field (scalar loop kwargs + top-level
                     wiring / infra).

Change size = number of characters in the edited surface (identical to 0b):
  * both sides strings -> max(len(old), len(new))
  * add/remove (one side None) -> len(str(existing side))
  * scalars -> len(str(changed value))

``project`` is deterministic: it walks the two configs in lockstep and, at every
leaf, keeps the candidate's value iff that leaf classifies into ``layer``,
otherwise reverts to the parent's value.  Additions (key only in candidate) are
kept only if they classify into ``layer``; deletions (key only in parent) are
honoured only if they classify into ``layer`` (else the parent value is kept).
"""
from __future__ import annotations

import re

LAYERS = ["configuration", "control", "action", "instruction"]

# Human-readable field list per layer — the single source of truth for the
# per-round brief constraint (§3.1) and for layers.md.  Keep in sync with
# ``classify_field``'s precedence rules below.
LAYER_FIELDS: dict[str, list[str]] = {
    "configuration": [
        "`processors[].history_n` / `history_mode` / `history_obs_clip`",
        "`processors[].loop_breaker_k` / `tried_action_mask`",
        "any other top-level scalar / wiring field not listed in the other layers",
    ],
    "control": [
        "fields inside a `processors[]` entry whose `_target_` references "
        "`processors.control.*` (mechanical post-processing hooks)",
    ],
    "action": [
        "`tools` / `tool_registry` / `actions` (new tools / action-space edits)",
    ],
    "instruction": [
        "`processors[].guidance`",
        "`processors[].system_builder.prompt`",
        "any other free-text `prompt` / `guidance` field",
    ],
}

_MISSING = object()


def walk_diff(old: dict, new: dict, prefix: str = ""):
    """Yield ``(path, old_leaf, new_leaf)`` for every changed leaf (add/remove/modify)."""
    keys = set(old) | set(new)
    for k in sorted(keys):
        p = f"{prefix}.{k}" if prefix else k
        in_old, in_new = k in old, k in new
        if in_old and in_new and isinstance(old[k], dict) and isinstance(new[k], dict):
            yield from walk_diff(old[k], new[k], p)
        elif in_old and in_new and isinstance(old[k], list) and isinstance(new[k], list):
            # list diff: recurse per index where both are dicts; report rest as leaf change
            for i in range(max(len(old[k]), len(new[k]))):
                pi = f"{p}[{i}]"
                oi = old[k][i] if i < len(old[k]) else None
                ni = new[k][i] if i < len(new[k]) else None
                if isinstance(oi, dict) and isinstance(ni, dict):
                    yield from walk_diff(oi, ni, pi)
                elif oi != ni:
                    yield (pi, oi, ni)
        elif in_old and in_new:
            if old[k] != new[k]:
                yield (p, old[k], new[k])
        else:
            yield (p, old[k] if in_old else None, new[k] if in_new else None)


def _processor_target_for(path: str, cfg: dict) -> str | None:
    """If the path is inside processors[i], return that entry's _target_ (from cfg)."""
    m = re.match(r"processors\[(\d+)\]", path)
    if not m:
        return None
    idx = int(m.group(1))
    procs = cfg.get("processors") or []
    if idx < len(procs) and isinstance(procs[idx], dict):
        return procs[idx].get("_target_", "")
    return None


def classify_field(path: str, parent_cfg: dict, candidate_cfg: dict) -> str:
    if re.search(r"\b(tools|tool_registry|actions)\b", path):
        return "action"
    tgt = _processor_target_for(path, candidate_cfg) or _processor_target_for(path, parent_cfg)
    if tgt and ".control." in tgt:
        return "control"
    if re.search(r"\b(prompt|guidance)\b", path):
        return "instruction"
    return "configuration"


def change_size(old, new) -> int:
    if isinstance(old, str) and isinstance(new, str):
        return max(len(old), len(new))
    if old is None or new is None:
        return len(str(new if old is None else old))
    return len(str(new))


def layers_of(parent_cfg: dict, candidate_cfg: dict) -> dict[str, int]:
    sizes = {layer: 0 for layer in LAYERS}
    for path, old, new in walk_diff(parent_cfg, candidate_cfg):
        layer = classify_field(path, parent_cfg, candidate_cfg)
        sizes[layer] += change_size(old, new)
    return sizes


def _ordered_keys(old: dict, new: dict) -> list:
    """Deterministic key order: parent's keys first, then candidate-only keys."""
    seen = []
    for k in old:
        seen.append(k)
    for k in new:
        if k not in seen:
            seen.append(k)
    return seen


def _project_node(old, new, prefix, parent_cfg, candidate_cfg, layer):
    """Recurse the two sides; return the projected node (or _MISSING to drop)."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = {}
        for k in _ordered_keys(old, new):
            p = f"{prefix}.{k}" if prefix else k
            v = _project_node(
                old.get(k, _MISSING), new.get(k, _MISSING),
                p, parent_cfg, candidate_cfg, layer,
            )
            if v is not _MISSING:
                out[k] = v
        return out
    if isinstance(old, list) and isinstance(new, list):
        out = []
        for i in range(max(len(old), len(new))):
            p = f"{prefix}[{i}]"
            v = _project_node(
                old[i] if i < len(old) else _MISSING,
                new[i] if i < len(new) else _MISSING,
                p, parent_cfg, candidate_cfg, layer,
            )
            if v is not _MISSING:
                out.append(v)
        return out

    # Leaf (scalar/string, or one side absent).
    lyr = classify_field(prefix, parent_cfg, candidate_cfg)
    if new is not _MISSING and lyr == layer:
        return new          # keep this layer's change
    if old is not _MISSING and lyr != layer:
        return old          # revert to parent (non-target layer)
    return _MISSING         # add outside layer / delete inside layer


def project(parent_cfg: dict, candidate_cfg: dict, layer: str) -> dict:
    """Clip ``candidate_cfg`` down to ``layer``: keep only that layer's changes,
    revert every other changed field to the parent's value. Deterministic."""
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r} (expected one of {LAYERS})")
    result = _project_node(parent_cfg, candidate_cfg, "", parent_cfg, candidate_cfg, layer)
    if result is _MISSING:
        return {}
    return result
