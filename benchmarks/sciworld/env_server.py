"""ScienceWorld environment server (SCIWORLD_SPEC.md §2.3).

Zero-dependency HTTP server (stdlib) that owns a fixed-size pool of ``N``
ScienceWorld workers.  Each worker is a *spawned* process holding one
``ScienceWorldEnv`` (its own JVM, ~250-290 MB, ~2 s to boot); the worker reuses
its env across episodes via ``env.load(...)``, so a JVM is never started per
episode.  Run inside the py3.8 ``ee-sciworld`` env::

    ee-sciworld/bin/python benchmarks/sciworld/env_server.py \
        --splits recipe/agent_evolver/envs/sciworld/splits.json --port 18091

Endpoints:
  GET  /health        -> {"status": "ok", "pool_size": int, "free": int}
  POST /create        -> {"env_id", "observation", "task_desc", "score", "done",
                          "possible_actions", "possible_objects", "moves",
                          "task_name", "variation"}
                        (body: {"episode_id": str})
  POST /step          -> {"observation", "reward", "done", "score", "in_valid",
                          "possible_actions", "possible_objects", "moves"}
                        (body: {"env_id": int, "action": str})
  POST /goal_progress -> {"goal_progress": str}   (body: {"env_id": int})
  POST /destroy       -> {"ok": true}             (body: {"env_id": int})

``in_valid`` is whether the *action* is an exact member of the pre-step valid
action-object combinations (used for the §7 metric, never shown to the agent).
The ambiguity-menu auto-answer (``0``) is driven by the client wrapper; from the
server's point of view it is just another ``/step``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SIMPLIFICATION = "teleportAction,openDoors,selfWateringFlowerPots,noElectricalAction"
ENV_STEP_LIMIT = 100
DEFAULT_POOL_SIZE = int(os.environ.get("SCIWORLD_POOL_SIZE", "16"))

# ── worker (spawned process) ─────────────────────────────────────────────────


def _state(env, observation, reward, done, info, *, in_valid=True, task_name=None, variation=None):
    possible_actions = [a for a in env.get_possible_actions() if a != "reset task"]
    return {
        "observation": observation,
        "reward": float(reward),
        "done": bool(done),
        "score": int(info["score"]),
        "in_valid": bool(in_valid),
        "possible_actions": list(possible_actions),
        "possible_objects": list(env.get_possible_objects()),
        "moves": int(info["moves"]),
        "task_name": task_name,
        "variation": variation,
    }


def _worker_loop(conn, simplification: str, env_step_limit: int) -> None:
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv(envStepLimit=env_step_limit)
    last_valid = []
    try:
        while True:
            try:
                cmd, payload = conn.recv()
            except EOFError:
                break
            if cmd == "reset":
                task_name, variation = payload
                env.load(task_name, variation, simplification)
                obs, info = env.reset()
                last_valid = list(info["valid"])
                conn.send({
                    **_state(env, obs, 0.0, False, info,
                             task_name=task_name, variation=variation),
                    "task_desc": env.get_task_description(),
                })
            elif cmd == "step":
                action = payload
                in_valid = action in last_valid
                obs, reward, done, info = env.step(action)
                last_valid = list(info["valid"])
                conn.send(_state(env, obs, reward, done, info, in_valid=in_valid))
            elif cmd == "goal_progress":
                conn.send({"goal_progress": env.get_goal_progress()})
            elif cmd == "close":
                conn.send({"ok": True})
                break
            else:
                conn.send({"error": f"unknown command {cmd!r}"})
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001 — best effort on teardown
            pass
        conn.close()


# ── pool (parent side) ───────────────────────────────────────────────────────


class _Worker:
    def __init__(self, conn, proc):
        self.conn = conn
        self.proc = proc
        self.busy = False
        self.env_id = None


class Pool:
    def __init__(self, size: int, simplification: str, env_step_limit: int):
        ctx = mp.get_context("spawn")
        self.workers: list[_Worker] = []
        for _ in range(size):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(target=_worker_loop, args=(child_conn, simplification, env_step_limit))
            proc.start()
            child_conn.close()
            self.workers.append(_Worker(parent_conn, proc))
        self.lock = threading.Lock()
        self._next_env_id = 0

    @property
    def size(self) -> int:
        return len(self.workers)

    def free_count(self) -> int:
        with self.lock:
            return sum(1 for w in self.workers if not w.busy)

    def _call(self, worker: _Worker, cmd, payload=None, timeout: float = 600.0):
        worker.conn.send((cmd, payload))
        if not worker.conn.poll(timeout):
            raise TimeoutError(f"worker {worker.env_id} timed out on {cmd!r}")
        return worker.conn.recv()

    def create(self, task_name: str, variation: int) -> tuple[int, dict]:
        with self.lock:
            worker = next((w for w in self.workers if not w.busy), None)
            if worker is None:
                raise RuntimeError("no free ScienceWorld worker")
            worker.busy = True
            worker.env_id = self._next_env_id
            self._next_env_id += 1
        try:
            result = self._call(worker, "reset", (task_name, variation))
        except Exception:
            self._release(worker)
            raise
        return worker.env_id, result

    def _find(self, env_id: int) -> _Worker | None:
        for w in self.workers:
            if w.busy and w.env_id == env_id:
                return w
        return None

    def step(self, env_id: int, action: str) -> dict:
        with self.lock:
            worker = self._find(env_id)
        if worker is None:
            raise KeyError(f"unknown env_id {env_id}")
        return self._call(worker, "step", action)

    def goal_progress(self, env_id: int) -> str:
        with self.lock:
            worker = self._find(env_id)
        if worker is None:
            raise KeyError(f"unknown env_id {env_id}")
        return self._call(worker, "goal_progress")["goal_progress"]

    def _release(self, worker: _Worker) -> None:
        with self.lock:
            worker.busy = False
            worker.env_id = None

    def destroy(self, env_id: int) -> bool:
        with self.lock:
            worker = self._find(env_id)
        if worker is None:
            return False
        self._release(worker)
        return True

    def shutdown(self) -> None:
        for w in self.workers:
            try:
                w.conn.send(("close", None))
                w.conn.recv()
            except Exception:  # noqa: BLE001
                pass
            try:
                w.proc.join(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            try:
                w.conn.close()
            except Exception:  # noqa: BLE001
                pass


# ── HTTP layer ───────────────────────────────────────────────────────────────

_pool: Pool | None = None
_splits: dict = {}


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
            self._send(200, {"status": "ok", "pool_size": _pool.size, "free": _pool.free_count()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")
        path = self.path.rstrip("/")

        try:
            if path == "/create":
                ep = _splits.get(data.get("episode_id", ""))
                if ep is None:
                    self._send(400, {"error": f"unknown episode_id {data.get('episode_id')!r}"})
                    return
                env_id, result = _pool.create(ep["task_name"], int(ep["variation"]))
                self._send(200, {"env_id": env_id, **result})
            elif path == "/step":
                result = _pool.step(int(data.get("env_id")), data.get("action", ""))
                self._send(200, result)
            elif path == "/goal_progress":
                self._send(200, {"goal_progress": _pool.goal_progress(int(data.get("env_id")))})
            elif path == "/destroy":
                _pool.destroy(int(data.get("env_id")))
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})
        except KeyError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *args):  # silence request logs
        pass


def main() -> None:
    global _pool, _splits
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--splits", required=True, help="path to envs/sciworld/splits.json")
    ap.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    args = ap.parse_args()

    splits = json.load(open(args.splits, encoding="utf-8"))
    _splits = splits["episodes"]
    _pool = Pool(args.pool_size, SIMPLIFICATION, ENV_STEP_LIMIT)

    print(f"[sciworld-env] pool={args.pool_size} episodes={len(_splits)} "
          f"on {args.host}:{args.port}", flush=True)
    try:
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    finally:
        _pool.shutdown()


if __name__ == "__main__":
    main()
