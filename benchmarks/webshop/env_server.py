"""WebShop (text mode) environment server.

Zero-dependency HTTP server (stdlib) that owns WebAgentTextEnv instances in-process.
Run inside the ee-webshop conda env (has Java for Lucene search + spacy)::

    WEBSHOP_ROOT=/mnt/llmshared-ssd-hd/hanguangzeng/verl-agent/agent_system/environments/env_package/webshop/webshop \
      /mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-webshop/bin/python \
      benchmarks/webshop/env_server.py --port 18081

Endpoints:
  GET  /health           -> {"status": "ok", "num_products": int, "num_goals": int}
  POST /create           -> {"env_id": str, "task_id": int, "observation": str,
                             "available_actions": {...}, "instruction": str}
  POST /step             -> {"observation": str, "reward": float, "done": bool,
                             "available_actions": {...}}
  POST /destroy          -> {"status": "ok"}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEBSHOP_ROOT = os.environ.get(
    "WEBSHOP_ROOT",
    "/mnt/llmshared-ssd-hd/hanguangzeng/verl-agent/agent_system/environments/env_package/webshop/webshop",
)
if WEBSHOP_ROOT not in sys.path:
    sys.path.insert(0, WEBSHOP_ROOT)

from web_agent_site.envs import WebAgentTextEnv  # noqa: E402


@dataclass
class EnvironmentRecord:
    env: "WebAgentTextEnv"
    session_prefix: str
    lock: threading.Lock = field(default_factory=threading.Lock)


class EnvironmentRegistry:
    def __init__(self, num_products, file_path=None, attr_path=None, human_goals=0, seed=0):
        self.num_products = num_products
        self.file_path = file_path
        self.attr_path = attr_path
        self.human_goals = human_goals
        self.seed = seed
        self._registry_lock = threading.Lock()
        self._records: dict[str, EnvironmentRecord] = {}
        self._bootstrap_env = self._make_env(seed, "bootstrap-")
        self._shared_server = self._bootstrap_env.server

    def _make_env(self, seed: int, session_prefix: str, server=None):
        kwargs = dict(
            observation_mode="text",
            num_products=self.num_products,
            human_goals=self.human_goals,
            seed=seed,
            session_prefix=session_prefix,
        )
        if self.file_path:
            kwargs["file_path"] = self.file_path
        if self.attr_path:
            kwargs["attr_path"] = self.attr_path
        if server is not None:
            kwargs["server"] = server
        return WebAgentTextEnv(**kwargs)

    @property
    def num_goals(self) -> int:
        return len(self._shared_server.goals)

    def create(self, task_id: int, seed: int) -> dict:
        if task_id < 0 or task_id >= self.num_goals:
            raise ValueError(f"task_id {task_id} is outside [0, {self.num_goals})")
        env_id = uuid.uuid4().hex
        session_prefix = f"hx-{env_id}-"
        env = self._make_env(seed, session_prefix, server=self._shared_server)
        observation, _ = env.reset(session=task_id)
        record = EnvironmentRecord(env=env, session_prefix=session_prefix)
        with self._registry_lock:
            self._records[env_id] = record
        return {
            "env_id": env_id,
            "task_id": task_id,
            "observation": observation,
            "available_actions": env.get_available_actions(),
            "instruction": env.get_instruction_text(),
        }

    def step(self, env_id: str, action: str) -> dict:
        record = self._get_record(env_id)
        with record.lock:
            observation, reward, done, info = record.env.step(action)
            available_actions = {} if done else record.env.get_available_actions()
        return {
            "observation": observation,
            "reward": float(reward),
            "done": bool(done),
            "info": dict(info or {}),
            "available_actions": available_actions,
        }

    def destroy(self, env_id: str) -> None:
        with self._registry_lock:
            record = self._records.pop(env_id, None)
        if record is None:
            return
        record.env.close()
        stale = [s for s in self._shared_server.user_sessions if s.startswith(record.session_prefix)]
        for s in stale:
            self._shared_server.user_sessions.pop(s, None)

    def _get_record(self, env_id: str) -> EnvironmentRecord:
        with self._registry_lock:
            record = self._records.get(env_id)
        if record is None:
            raise KeyError(f"unknown env_id {env_id}")
        return record


_registry: EnvironmentRegistry | None = None


def _parse_num_products():
    v = os.environ.get("WEBSHOP_NUM_PRODUCTS", "1000").strip()
    return None if v in ("", "full", "none") else int(v)


def _get_registry() -> EnvironmentRegistry:
    global _registry
    if _registry is None:
        _registry = EnvironmentRegistry(
            num_products=_parse_num_products(),
            file_path=os.environ.get("WEBSHOP_FILE_PATH") or None,
            attr_path=os.environ.get("WEBSHOP_ATTR_PATH") or None,
            human_goals=int(os.environ.get("WEBSHOP_HUMAN_GOALS", "0")),
            seed=int(os.environ.get("WEBSHOP_SEED", "0")),
        )
    return _registry


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") == "/health":
            reg = _get_registry()
            self._send(200, {"status": "ok", "num_products": reg.num_products, "num_goals": reg.num_goals})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")
        reg = _get_registry()
        path = self.path.rstrip("/")

        try:
            if path == "/create":
                self._send(200, reg.create(int(data["task_id"]), int(data.get("seed", 0))))
            elif path == "/step":
                self._send(200, reg.step(data["env_id"], data["action"]))
            elif path == "/destroy":
                reg.destroy(data["env_id"])
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})
        except (ValueError, KeyError) as exc:
            self._send(400, {"error": str(exc)})

    def log_message(self, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18081)
    args = ap.parse_args()
    reg = _get_registry()
    print(f"[webshop-env] num_products={reg.num_products} num_goals={reg.num_goals} on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
