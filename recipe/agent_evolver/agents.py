# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Agent runners for ALFWorld and WebShop.

Each runner drives a plain ReAct loop over the benchmark env server with the
given HarnessSpec and the weak agent model (Qwen3.5-4B) served by vLLM via
HarnessX's LiteLLMProvider.

The prompt templates and input construction (task description + recent
observation/action history + admissible actions, with <think>/<action> tags)
are taken verbatim from Skill1 / verl-agent so the base model reproduces its
step-0 success rate instead of the previous stateless 0%.

Model weights are never modified here.
"""
from __future__ import annotations

import asyncio
import os

import httpx

from harnessx.core.events import Message
from harnessx.providers.litellm_provider import LiteLLMProvider

from . import processors
from .spec import (
    ALF_MAX_STEPS,
    SCIWORLD_MAX_STEPS,
    TEXTCRAFT_MAX_STEPS,
    WS_MAX_STEPS,
    HarnessSpec,
)

ALFWORLD_ENV_URL = os.environ.get("EVOLVER_ALFWORLD_ENV_URL", "http://127.0.0.1:18082")
WEBSHOP_ENV_URL = os.environ.get("EVOLVER_WEBSHOP_ENV_URL", "http://127.0.0.1:18081")


async def _complete(provider: LiteLLMProvider, system: str, user: str) -> str:
    msgs = [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]
    resp = await provider.complete(msgs, [])
    return resp.content


# ── Prompt templates (Skill1 / verl-agent, skills slot empty for base) ────────

ALFWORLD_TEMPLATE_NO_HIS = """\
You are an expert agent operating in the ALFRED Embodied Environment.
{skills}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

ALFWORLD_TEMPLATE = """\
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
{skills}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

WEBSHOP_TEMPLATE_NO_HIS = """\
You are an expert autonomous agent operating in the WebShop e-commerce environment.
{skills}
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

WEBSHOP_TEMPLATE = """\
You are an expert autonomous agent operating in the WebShop e-commerce environment.
{skills}
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

TEXTCRAFT_TEMPLATE_NO_HIS = """\
You are an expert agent crafting items in a Minecraft-like text environment. Your task is to: craft {goal}.
Here are the crafting commands you can use:
{commands_text}
Rules: always give a quantity, e.g. "get 1 lapis lazuli" or "craft 1 golden carrot using 8 gold nugget, 1 carrot". Do not say whether an item comes from your inventory or the environment. Use only the recipes given above.
{skills}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

TEXTCRAFT_TEMPLATE = """\
You are an expert agent crafting items in a Minecraft-like text environment. Your task is to: craft {goal}.
Here are the crafting commands you can use:
{commands_text}
Rules: always give a quantity, e.g. "get 1 lapis lazuli" or "craft 1 golden carrot using 8 gold nugget, 1 carrot". Do not say whether an item comes from your inventory or the environment. Use only the recipes given above.
{skills}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

SCIWORLD_TEMPLATE_NO_HIS = """\
You are an expert agent operating in the ScienceWorld environment, a text-based simulation of science experiments. Your task is to: {task_description}
{skills}
Your current observation is: {current_observation}
Valid action templates (replace OBJ and LOC with objects or locations): {action_templates}
Objects you can currently see: {visible_objects}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose one valid action for current step and present it within <action> </action> tags."""

SCIWORLD_TEMPLATE = """\
You are an expert agent operating in the ScienceWorld environment, a text-based simulation of science experiments. Your task is to: {task_description}
{skills}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Valid action templates (replace OBJ and LOC with objects or locations): {action_templates}
Objects you can currently see: {visible_objects}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose one valid action for current step and present it within <action> </action> tags."""


# ── Action extraction / formatting (mirrors Skill1 projection + env_manager) ──

def extract_action(raw: str) -> str:
    """Return the payload of <action>...</action>, lowercased like the reference
    projection. On a malformed response return the raw tail — the env treats it
    as an invalid action and applies its penalty (we do NOT force-fallback)."""
    low = raw.lower()
    start = low.find("<action>")
    end = low.find("</action>")
    if start != -1 and end != -1 and start < end:
        return low[start + len("<action>"): end].strip()
    return raw[-30:].strip()


def _ws_action_list(avail: dict) -> list[str]:
    actions = []
    if avail.get("has_search_bar"):
        actions.append("search[<your query>]")
    actions.extend(f"click[{txt}]" for txt in avail.get("clickables", []))
    return actions


def _fmt_ws_available(avail: dict) -> str:
    return "\n".join(f"'{s}'," for s in _ws_action_list(avail))


def _ws_is_valid(action: str, avail: dict) -> bool:
    if action.startswith("search["):
        return bool(avail.get("has_search_bar")) and len(action) > len("search[]")
    if action.startswith("click["):
        value = action[len("click["): -1].strip()
        return value in {str(c).lower() for c in avail.get("clickables", [])}
    return False


# ── ALFWorld ────────────────────────────────────────────────────────────────

# Errors where the connection was never established → the request never reached
# the server, so retrying is always safe. Errors where the connection dropped
# mid-response (ReadError/ReadTimeout/RemoteProtocolError) are excluded from the
# connect-only set: the action may already have been applied server-side.
_CONNECT_ONLY = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
_RETRY_ALL = (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError,
              httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout)


async def _post_retry(client: httpx.AsyncClient, url: str, *, json=None, retries: int = 6,
                      connect_only: bool = False):
    """POST with retry on transient connection errors.

    ``/reset``/``/create`` are idempotent, so all transient errors are retried.
    ``/step`` retries only ``connect_only`` (connection never established); when
    the response is lost mid-flight the action may already have been applied
    server-side and retrying would double-step, so that case is NOT retried.
    """
    retryable = _CONNECT_ONLY if connect_only else _RETRY_ALL
    last: BaseException | None = None
    for i in range(retries):
        try:
            return await client.post(url, json=json)
        except retryable as exc:
            last = exc
            await asyncio.sleep(2.0 * (i + 1))
    raise last  # type: ignore[misc]


async def run_alfworld(
    provider: LiteLLMProvider,
    spec: HarnessSpec,
    game_idxs: list[int],
    max_steps: int = ALF_MAX_STEPS,
    seed: int = 1234,
    env_url: str | None = None,
) -> list[dict]:
    records: list[dict] = []
    skills = spec.guidance
    async with httpx.AsyncClient(base_url=env_url or ALFWORLD_ENV_URL, timeout=180.0) as client:
        for gi in game_idxs:
            r = (await _post_retry(client, "/reset", json={"game_idx": gi, "seed": seed})).json()
            obs = r["observation"]
            adm = r.get("admissible_commands", [])
            task_desc = obs.split("Your task is to: ", 1)[1].strip() if "Your task is to: " in obs else obs
            history: list[tuple[str, str, bool]] = []
            done = False
            steps = 0
            won = False
            traj: list[dict] = []
            final_obs = obs

            while not done and steps < max_steps:
                pre_obs = obs
                base_show = [s for s in adm if s != "help"]
                adm_show = processors.mask_admissible(history, base_show, spec.processors)
                adm_fmt = "\n ".join(f"'{s}'" for s in adm_show)
                action_history, shown_n = processors.render_history(history, spec.processors)
                if not history:
                    user = ALFWORLD_TEMPLATE_NO_HIS.format(
                        skills=skills, current_observation=pre_obs, admissible_actions=adm_fmt)
                else:
                    user = ALFWORLD_TEMPLATE.format(
                        task_description=task_desc, skills=skills,
                        step_count=len(history), history_length=shown_n,
                        action_history=action_history,
                        current_step=len(history) + 1,
                        current_observation=pre_obs, admissible_actions=adm_fmt)

                raw = await _complete(provider, spec.system_prompt, user)
                action = extract_action(raw)
                valid = action in adm
                traj.append({
                    "step": steps + 1, "prompt": user, "obs": pre_obs, "adm": adm_show,
                    "raw": raw, "action": action, "invalid": not valid,
                })

                try:
                    r = (await client.post("/step", json={"action": action})).json()
                except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError,
                        httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as exc:
                    # /step 偶发断连：动作可能已执行但响应丢失，无法安全重试。
                    # 标记当前 game 失败并跳过（下一个 /reset 会重新 seed+skip）。
                    print(f"[alfworld] game {gi}: STEP-CONN-ERROR at step {steps + 1}: "
                          f"{type(exc).__name__}", flush=True)
                    won = False
                    break
                obs = r["observation"]
                adm = r.get("admissible_commands", [])
                won = bool(r.get("won"))
                done = bool(r.get("done"))
                history.append((pre_obs, action, valid))
                steps += 1
                final_obs = obs

            records.append({
                "task_id": gi,
                "success": won,
                "reward": 1.0 if won else 0.0,
                "steps": steps,
                "goal": task_desc,
                "trajectory": traj,
                "final_obs": final_obs,
            })
    return records


# ── WebShop ─────────────────────────────────────────────────────────────────

async def run_webshop(
    provider: LiteLLMProvider,
    spec: HarnessSpec,
    task_ids: list[int],
    max_steps: int = WS_MAX_STEPS,
    seed: int = 0,
    env_url: str | None = None,
) -> list[dict]:
    records: list[dict] = []
    skills = spec.guidance
    async with httpx.AsyncClient(base_url=env_url or WEBSHOP_ENV_URL, timeout=180.0) as client:
        for tid in task_ids:
            c = (await _post_retry(client, "/create", json={"task_id": tid, "seed": seed})).json()
            env_id = c["env_id"]
            obs = c["observation"]
            avail = c["available_actions"]
            task_desc = c.get("instruction", "")
            history: list[tuple[str, str, bool]] = []
            done = False
            steps = 0
            reward = 0.0
            reward_info: dict | None = None
            traj: list[dict] = []
            final_obs = obs

            while not done and steps < max_steps:
                pre_obs = obs
                avail_list = processors.mask_admissible(history, _ws_action_list(avail), spec.processors)
                avail_fmt = "\n".join(f"'{s}'," for s in avail_list)
                action_history, shown_n = processors.render_history(history, spec.processors)
                if not history:
                    user = WEBSHOP_TEMPLATE_NO_HIS.format(
                        skills=skills, task_description=task_desc,
                        current_observation=pre_obs, available_actions=avail_fmt)
                else:
                    user = WEBSHOP_TEMPLATE.format(
                        skills=skills, task_description=task_desc,
                        step_count=len(history), history_length=shown_n,
                        action_history=action_history,
                        current_step=len(history) + 1,
                        current_observation=pre_obs, available_actions=avail_fmt)

                raw = await _complete(provider, spec.system_prompt, user)
                action = extract_action(raw)
                valid = _ws_is_valid(action, avail)
                traj.append({
                    "step": steps + 1, "prompt": user, "obs": pre_obs, "avail": avail_fmt,
                    "raw": raw, "action": action, "invalid": not valid,
                })

                r = (await _post_retry(client, "/step", json={"env_id": env_id, "action": action},
                                        connect_only=True)).json()
                obs = r["observation"]
                avail = r.get("available_actions", {})
                reward = float(r["reward"])
                done = bool(r["done"])
                if done and r.get("reward_info") is not None:
                    reward_info = r["reward_info"]
                history.append((pre_obs, action, valid))
                steps += 1
                final_obs = obs

            await client.post("/destroy", json={"env_id": env_id})
            records.append({
                "task_id": tid,
                "success": reward >= 0.999,
                "reward": reward,
                "reward_info": reward_info,
                "steps": steps,
                "goal": task_desc,
                "trajectory": traj,
                "final_obs": final_obs,
            })
    return records


# ── TextCraft ────────────────────────────────────────────────────────────────

async def run_textcraft(
    provider: LiteLLMProvider,
    spec: HarnessSpec,
    task_ids: list[str],
    max_steps: int = TEXTCRAFT_MAX_STEPS,
    seed: int = 0,
    env_url: str | None = None,
) -> list[dict]:
    """In-process TextCraft rollout (no env server; SPEC §1/§3).

    A single shared CraftingTree + vendored TextCraftEnv are built once per
    worker and reused across episodes. ``seed``/``env_url`` are unused and kept
    only for signature symmetry with ``run_alfworld``/``run_webshop``.
    """
    from .envs.textcraft.env import VENDOR_DIR, TextCraftEnvWrapper, load_manifest
    from .envs.textcraft.vendor.crafting_tree import CraftingTree
    from .envs.textcraft import failure_labels, projection

    tree = CraftingTree(VENDOR_DIR)
    manifest = load_manifest()
    wrapper = TextCraftEnvWrapper(manifest=manifest, tree=tree)
    records: list[dict] = []
    skills = spec.guidance

    for tid in task_ids:
        row = manifest[tid]
        task_desc = row["goal"]
        commands_text = row["commands_text"]
        obs, _info = wrapper.reset(tid)
        history: list[tuple[str, str, bool]] = []
        done = False
        steps = 0
        won = False
        reward = 0.0
        traj: list[dict] = []
        final_obs = obs

        while not done and steps < max_steps:
            pre_obs = obs
            adm_show = wrapper.admissible
            adm_fmt = ", ".join(f"'{s}'" for s in adm_show)
            action_history, shown_n = processors.render_history(history, spec.processors)
            if not history:
                user = TEXTCRAFT_TEMPLATE_NO_HIS.format(
                    goal=task_desc, commands_text=commands_text, skills=skills,
                    current_observation=pre_obs, admissible_actions=adm_fmt)
            else:
                user = TEXTCRAFT_TEMPLATE.format(
                    goal=task_desc, commands_text=commands_text, skills=skills,
                    step_count=len(history), history_length=shown_n,
                    action_history=action_history, current_step=len(history) + 1,
                    current_observation=pre_obs, admissible_actions=adm_fmt)

            raw = await _complete(provider, spec.system_prompt, user)
            raw_action, action = projection.extract(raw)
            obs, reward, done, info = wrapper.step(action)
            valid = action != ""
            traj.append({
                "step": steps + 1, "prompt": user, "obs": pre_obs,
                "raw": raw, "raw_action": raw_action, "action": action,
                "invalid": not valid,
                "env_message": info["env_message"], "inventory": info["inventory"],
                "in_admissible": info["in_admissible"],
            })
            history.append((pre_obs, action, valid))
            steps += 1
            final_obs = obs
            won = reward >= 0.999

        lab = failure_labels.label_trajectory(tree, row, traj, won)
        records.append({
            "task_id": tid,
            "success": won,
            "reward": 1.0 if won else 0.0,
            "steps": steps,
            "goal": task_desc,
            "trajectory": traj,
            "final_obs": final_obs,
            "frontmatter_extra": {
                "depth": row["depth"],
                "episode_label": lab["episode_label"],
                "progress": lab["progress"],
            },
        })
    return records


# ── ScienceWorld ──────────────────────────────────────────────────────────────

async def run_sciworld(
    provider: LiteLLMProvider,
    spec: HarnessSpec,
    episode_ids: list[str],
    max_steps: int = SCIWORLD_MAX_STEPS,
    seed: int = 0,
    env_url: str | None = None,
) -> list[dict]:
    """ScienceWorld rollout over the py3.8 env-server pool (SPEC §8.2).

    One :class:`ScienceWorldClient` per worker (episodes processed serially);
    the blocking HTTP calls run in a thread so the async worker keeps the event
    loop responsive.  ``seed`` is unused (the split is fixed) and kept only for
    signature symmetry.
    """
    from .envs.sciworld.env import SCIWORLD_ENV_URL, ScienceWorldClient
    from .envs.sciworld import failure_labels, projection

    client = ScienceWorldClient(env_url or SCIWORLD_ENV_URL)
    records: list[dict] = []
    skills = spec.guidance
    try:
        for eid in episode_ids:
            obs, _info = await asyncio.to_thread(client.reset, eid)
            task_desc = client.task_desc
            task_name = client.task_name
            variation = client.variation
            history: list[tuple[str, str, bool]] = []
            done = False
            steps = 0
            traj: list[dict] = []
            final_obs = obs

            while not done and steps < max_steps:
                pre_obs = obs
                action_templates = "\n".join(f"- {a}" for a in client.possible_actions)
                visible_objects = ", ".join(client.possible_objects)
                action_history, shown_n = processors.render_history(history, spec.processors)
                if not history:
                    user = SCIWORLD_TEMPLATE_NO_HIS.format(
                        task_description=task_desc, skills=skills,
                        current_observation=pre_obs,
                        action_templates=action_templates, visible_objects=visible_objects)
                else:
                    user = SCIWORLD_TEMPLATE.format(
                        task_description=task_desc, skills=skills,
                        step_count=len(history), history_length=shown_n,
                        action_history=action_history, current_step=len(history) + 1,
                        current_observation=pre_obs,
                        action_templates=action_templates, visible_objects=visible_objects)

                raw = await _complete(provider, spec.system_prompt, user)
                raw_action, action = projection.extract(raw)
                if action == "":
                    # Parse failure: don't call the env; fixed observation, one step.
                    obs = "Invalid action format."
                    info = {"score_raw": client.score_raw, "score": client.score,
                            "in_valid": False, "auto_disambiguated": False, "moves": 0}
                    done = False
                else:
                    obs, _reward, done, info = await asyncio.to_thread(client.step, action)
                traj.append({
                    "step": steps + 1, "prompt": user, "obs": pre_obs,
                    "raw": raw, "raw_action": raw_action, "action": action,
                    "invalid": action == "",
                    "env_message": obs,
                    "score": info["score_raw"],
                    "in_valid": bool(info["in_valid"]),
                    "auto_disambiguated": bool(info["auto_disambiguated"]),
                })
                history.append((pre_obs, action, action != ""))
                steps += 1
                final_obs = obs

            success = client.score_raw == 100
            goal_progress = await asyncio.to_thread(client.goal_progress)
            lab = failure_labels.label_trajectory(traj, goal_progress, success)
            records.append({
                "task_id": eid,
                "success": success,
                "reward": client.score,
                "steps": steps,
                "goal": task_desc,
                "trajectory": traj,
                "final_obs": final_obs,
                "task_name": task_name,
                "variation": variation,
                "frontmatter_extra": {
                    "task_name": task_name,
                    "variation": variation,
                    "score": client.score,
                    "episode_label": lab["episode_label"],
                    "first_unmet_subgoal": lab["first_unmet_subgoal"] or "",
                },
            })
            await asyncio.to_thread(client.destroy)
    finally:
        client.close()
    return records
