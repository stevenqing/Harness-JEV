# Vendored ScienceWorld reference

## Source

The spec (SCIWORLD_SPEC.md §1) names Tencent's RLVMR repo
(https://github.com/Tencent/DigitalHuman/tree/main/RLVMR — a verl-agent fork)
as the starting point for the environment wrapper.  RLVMR itself is not
reachable from this host (no egress), so the files were taken from the local
verl-agent checkout at `/mnt/llmshared-ssd-hd/hanguangzeng/verl-agent`, commit
`20bd331bdbc9026a5668e11362178e10ab7400c8` (2026-06-09), which is the same
lineage:

- `code/agent_system/environments/env_package/sciworld/envs.py`
  → `envs.py` (verbatim)
- `code/agent_system/environments/env_package/sciworld/projection.py`
  → `projection.py` (verbatim)
- `code/agent_system/environments/prompts/sciworld.py`
  → `prompts_scienceworld.py` (verbatim)

License: Apache-2.0 (see `LICENSE`).

## Why these files

They document the `ScienceWorldEnv` load/step/result shape (`envs.py`), the
reference action projection, and the reference prompt.  They are **reference
only**: the real harness re-implements the SPEC-mandated behaviour in the parent
package (`envs/sciworld/{env,projection,failure_labels}.py` + the
`SCIWORLD_TEMPLATE` in `agents.py`), because the spec deliberately diverges from
this vendored code:

- prompt uses `<think>/<action>` tags (spec §5), not `Thought:`/`Action:` (the
  vendored `prompts_scienceworld.py` uses the latter);
- projection does no case conversion and no `.rstrip(".")`, and does not reject
  CJK (spec §6);
- the wrapper is an HTTP client against `benchmarks/sciworld/env_server.py`,
  not a Ray actor (`envs.py` is Ray-backed).

## Changes

None — the files are verbatim copies for provenance.  The actual implementation
lives outside `vendor/`.
