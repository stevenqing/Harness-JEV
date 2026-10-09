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
    AGENT_API_BASE,
    AGENT_ENABLE_THINKING,
    AGENT_MAX_TOKENS,
    AGENT_MODEL,
    AGENT_TEMPERATURE,
    AGENT_TOP_P,
    ALF_MAX_STEPS,
    WS_MAX_STEPS,
    HarnessSpec,
)

ALFWORLD_ENV_URL = os.environ.get("EVOLVER_ALFWORLD_ENV_URL", "http://127.0.0.1:18082")
WEBSHOP_ENV_URL = os.environ.get("EVOLVER_WEBSHOP_ENV_URL", "http://127.0.0.1:18081")


def make_agent_provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model=AGENT_MODEL,
        api_base=AGENT_API_BASE,
        api_key="EMPTY",
        temperature=AGENT_TEMPERATURE,
        top_p=AGENT_TOP_P,
        max_tokens=AGENT_MAX_TOKENS,
        extra_body={"chat_template_kwargs": {"enable_thinking": AGENT_ENABLE_THINKING}},
    )


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
                history.append((pre_obs, action, valid))
                steps += 1
                final_obs = obs

            await client.post("/destroy", json={"env_id": env_id})
            records.append({
                "task_id": tid,
                "success": reward >= 0.999,
                "reward": reward,
                "steps": steps,
                "goal": task_desc,
                "trajectory": traj,
                "final_obs": final_obs,
            })
    return records
