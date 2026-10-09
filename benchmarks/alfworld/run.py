#!/usr/bin/env python3
"""Run Qwen3.5-4B on ALFWorld through HarnessX's provider layer.

Drives a ReAct loop: env reset -> text observation -> model picks one <action>
-> env.step, until done/won or max-steps. The model is served by vLLM at
``--api-base`` (OpenAI-compatible) and called via HarnessX's LiteLLMProvider.

Usage (inside the HarnessX .venv)::

    .venv/bin/python benchmarks/alfworld/run.py --num-tasks 20
"""
from __future__ import annotations

import argparse
import asyncio
import re

import httpx

from harnessx.core.events import Message
from harnessx.providers.litellm_provider import LiteLLMProvider

SYSTEM_PROMPT = (
    "You are an agent in a household, solving a task step by step.\n"
    "Given the observation and the list of admissible actions, reply with the single best admissible action for the next step.\n"
    "Reply with only the action text and nothing else. Choose only from the listed admissible actions. Do not invent actions."
)

ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)


def _answer(text: str) -> str:
    """The real answer follows the final </think> delimiter (thinking-mode models)."""
    return text.rsplit("</think>", 1)[1] if "</think>" in text else text


def parse_action(text: str) -> str | None:
    m = ACTION_RE.search(text)
    return m.group(1).strip().lower() if m else None


def pick_action(text: str, adm: list[str]) -> str:
    """Choose the model's action, validating against the admissible set."""
    for portion in (_answer(text), text):
        a = parse_action(portion)
        if a and a in adm:
            return a
    low = text.lower()
    for cmd in adm:
        if cmd in low:
            return cmd
    return adm[0] if adm else "look"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/Qwen3.5-4B")
    ap.add_argument("--api-base", default="http://127.0.0.1:8200/v1")
    ap.add_argument("--env-url", default="http://127.0.0.1:18082")
    ap.add_argument("--num-tasks", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=50)
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

    won = 0
    async with httpx.AsyncClient(base_url=args.env_url, timeout=180.0) as client:
        health = (await client.get("/health")).json()
        print(f"[alfworld] env health: {health}", flush=True)

        for i in range(args.num_tasks):
            r = (await client.post("/reset")).json()
            obs = r["observation"]
            done = False
            steps = 0
            while not done and steps < args.max_steps:
                adm = r.get("admissible_commands", [])
                listing = "\n".join(f"- {a}" for a in adm)
                msgs = [
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=f"Observation:\n{obs}\n\nAdmissible actions:\n{listing}"),
                ]
                resp = await provider.complete(msgs, [])
                action = pick_action(resp.content, adm)
                r = (await client.post("/step", json={"action": action})).json()
                obs = r["observation"]
                done = bool(r["done"])
                steps += 1
            won_i = bool(r.get("won"))
            won += int(won_i)
            print(f"[alfworld] task {i}: won={won_i} steps={steps}", flush=True)

    print(f"\nALFWorld success rate: {won}/{args.num_tasks} = {won / args.num_tasks:.1%}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
