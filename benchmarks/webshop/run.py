#!/usr/bin/env python3
"""Run Qwen3.5-4B on WebShop through HarnessX's provider layer.

Drives a ReAct loop: env /create -> text observation -> model picks search[...]/click[...]
-> env /step, until done. Reward >= 0.999 counts as success. The model is served by
vLLM at ``--api-base`` and called via HarnessX's LiteLLMProvider.

Usage (inside the HarnessX .venv)::

    .venv/bin/python benchmarks/webshop/run.py --num-tasks 20
"""
from __future__ import annotations

import argparse
import asyncio
import re

import httpx

from harnessx.core.events import Message
from harnessx.providers.litellm_provider import LiteLLMProvider

SYSTEM_PROMPT = (
    "You are an autonomous shopping agent in the WebShop text environment.\n"
    "Given the observation and the available actions, reply with the single best next action.\n"
    "Valid actions have one of these forms:\n"
    "- search[keywords]\n"
    "- click[value]\n"
    "Reply with only the action text and nothing else. Do not invent clickable values. Use only the available actions shown."
)

ACTION_RE = re.compile(r"<action>\s*((?:search|click)\[[^\n\]]+\])\s*</action>", re.IGNORECASE)
FALLBACK_RE = re.compile(r"((?:search|click)\[[^\n\]]+\])", re.IGNORECASE)


def _format_actions(avail: dict) -> str:
    actions = []
    if avail.get("has_search_bar"):
        actions.append("search[keywords]")
    actions.extend(f"click[{v}]" for v in avail.get("clickables", []) if v != "search")
    return "\n".join(f"- {a}" for a in actions) or "- no valid actions"


def _format_obs(obs: str, avail: dict) -> str:
    return f"Observation:\n{obs}\n\nAvailable actions:\n{_format_actions(avail)}"


def _answer(text: str) -> str:
    """The real answer follows the final </think> delimiter (thinking-mode models)."""
    return text.rsplit("</think>", 1)[1] if "</think>" in text else text


def extract_action(text: str) -> str | None:
    m = ACTION_RE.search(_answer(text))
    if not m:
        m = ACTION_RE.search(text)
    if not m:
        m = FALLBACK_RE.search(_answer(text))
    if not m:
        m = FALLBACK_RE.search(text)
    return m.group(1).strip().lower() if m else None


def is_valid_action(action: str | None, avail: dict) -> bool:
    if action is None:
        return False
    if action.startswith("search["):
        return bool(avail.get("has_search_bar")) and len(action) > len("search[]")
    if not action.startswith("click["):
        return False
    value = action[len("click[") : -1].strip().lower()
    clickables = {str(c).lower() for c in avail.get("clickables", [])}
    return value in clickables and value != "search"


def fallback_action(avail: dict) -> str:
    clickables = [str(c) for c in avail.get("clickables", []) if str(c).lower() != "search"]
    if clickables:
        return f"click[{clickables[0]}]"
    return "search[item]"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/Qwen3.5-4B")
    ap.add_argument("--api-base", default="http://127.0.0.1:8200/v1")
    ap.add_argument("--env-url", default="http://127.0.0.1:18081")
    ap.add_argument("--num-tasks", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    provider = LiteLLMProvider(
        model=args.model,
        api_base=args.api_base,
        api_key="EMPTY",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    success = 0
    total_reward = 0.0
    async with httpx.AsyncClient(base_url=args.env_url, timeout=180.0) as client:
        health = (await client.get("/health")).json()
        print(f"[webshop] env health: {health}", flush=True)

        for task_id in range(args.num_tasks):
            c = (await client.post("/create", json={"task_id": task_id, "seed": 0})).json()
            env_id = c["env_id"]
            obs = c["observation"]
            avail = c["available_actions"]
            done = False
            steps = 0
            reward = 0.0
            while not done and steps < args.max_steps:
                msgs = [
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=_format_obs(obs, avail)),
                ]
                resp = await provider.complete(msgs, [])
                action = extract_action(resp.content)
                if not is_valid_action(action, avail):
                    action = fallback_action(avail)
                r = (await client.post("/step", json={"env_id": env_id, "action": action})).json()
                obs = r["observation"]
                avail = r["available_actions"]
                reward = float(r["reward"])
                done = bool(r["done"])
                steps += 1
            await client.post("/destroy", json={"env_id": env_id})
            ok = reward >= 0.999
            success += int(ok)
            total_reward += reward
            print(f"[webshop] task {task_id}: reward={reward:.3f} success={ok} steps={steps}", flush=True)

    print(f"\nWebShop success: {success}/{args.num_tasks} = {success / args.num_tasks:.1%} "
          f"(avg reward {total_reward / args.num_tasks:.3f})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
