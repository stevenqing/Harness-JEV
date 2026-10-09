"""ALFWorld (TextWorld) environment server.

Zero-dependency HTTP server (stdlib) that owns a single AlfredTWEnv (batch_size=1).
Run inside the ee-alfworld conda env which has textworld + alfworld installed::

    ALFWORLD_DATA=/mnt/llmshared-ssd-hd/cty/data/alfworld \
      /mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-alfworld/bin/python \
      benchmarks/alfworld/env_server.py --port 18082

Endpoints:
  GET  /health  -> {"status": "ok", "num_games": int, "game_idx": int}
  POST /reset   -> {"observation": str, "admissible_commands": [...], "game_idx": int}
                 (body: {"game_idx": int, "seed": int} — seed defaults to 1234)
  POST /step    -> {"observation": str, "won": bool, "done": bool, "admissible_commands": [...]}
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import yaml

import alfworld.agents.environment as environment

CONFIG_PATH = os.environ.get(
    "ALFWORLD_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_tw.yaml"),
)
SPLIT = os.environ.get("ALFWORLD_SPLIT", "eval_in_distribution")
# Opt-in: expose the hand-coded expert's next action as "expert_plan" on /reset
# and /step. Off by default so the official-eval servers are byte-for-byte the
# old behavior. offline_diag/collect.py runs its servers with this on.
EXPERT_PLAN = os.environ.get("ALFWORLD_EXPERT_PLAN", "0") == "1"

_tw = None   # AlfredTWEnv wrapper (num_games)
_env = None  # gym env from init_env(batch_size=1)
_game_idx = 0


def _load_env():
    global _tw, _env
    if _env is not None:
        return _env
    config = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    config["env"]["type"] = "AlfredTWEnv"
    config["general"]["use_cuda"] = False
    # Override the expert type (default "handcoded" -> next action only). The
    # PLANNER expert sets extra.expert_plan to the FULL policy_commands plan, which
    # offline_diag §5 needs to recover the target's true container.
    if os.environ.get("ALFWORLD_EXPERT_TYPE"):
        config["env"]["expert_type"] = os.environ["ALFWORLD_EXPERT_TYPE"]
    _tw = getattr(environment, "AlfredTWEnv")(config, train_eval=SPLIT)
    if EXPERT_PLAN:
        # AlfredTWEnv.init_env only wraps the env in AlfredExpert (which computes
        # state["extra.expert_plan"] via the hand-coded expert) when
        # train_eval == "train" and training_method == "dagger". Game files were
        # already collected for the real SPLIT above (collect_game_files runs in
        # __init__), and init_env reads train_eval only for (a) domain
        # randomization, which the config already sets False, and (b) the
        # expert_plan gate. Flip it transiently to force the wrapper, then restore.
        saved = _tw.train_eval
        _tw.train_eval = "train"
        try:
            _env = _tw.init_env(batch_size=1)
        finally:
            _tw.train_eval = saved
    else:
        _env = _tw.init_env(batch_size=1)
    return _env


def _first(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def _expert_plan(infos):
    """Next expert action from batched infos, or None.

    With the AlfredExpert wrapper, infos carries "extra.expert_plan" = a list of
    per-game plans, e.g. [["go to cabinet 1"]] (batch dim -> inner list). Unnest
    both levels to the single string. Returns None when the field is absent or
    empty (the spec records those steps as null).
    """
    ep = infos.get("extra.expert_plan") if isinstance(infos, dict) else None
    ep = _first(ep)  # drop batch dim -> ["go to cabinet 1"]
    ep = _first(ep)  # drop plan list   -> "go to cabinet 1"
    return ep if ep else None


def _expert_plan_full(infos):
    """Full remaining expert plan (list of actions), or [].

    Same source as ``_expert_plan`` but only drops the batch dim, keeping the
    whole plan list. Used by offline_diag §5 to recover the target object's true
    container ("first go to X before first take target").
    """
    ep = infos.get("extra.expert_plan") if isinstance(infos, dict) else None
    ep = _first(ep)  # drop batch dim -> ["go to cabinet 1", "take mug 1 ...", ...]
    return list(ep) if ep else []


def _gamefile(infos):
    """Absolute path to this game's traj_data.json, or None.

    Always requested (extras=["gamefile"]) and set by the AlfredInfos wrapper as
    ``state["extra.gamefile"]``, so it is present regardless of expert type.
    offline_diag §5 reads this JSON's plan.high_pddl to recover the target
    object's true container (the planner expert's policy_commands is not
    populated in this build).
    """
    gf = infos.get("extra.gamefile") if isinstance(infos, dict) else None
    return _first(gf) if gf else None


def _gamefiles_order(seed: int) -> list[str]:
    """Seed-ordered gamefile paths, mirroring ``env.seed(seed)``.

    ``textworld_batch.seed`` shuffles ``list(self.gamefiles)`` with
    ``numpy.random.RandomState(seed)``; game_idx then indexes this order. We
    expose the same order so offline_diag can map task_id -> gamefile in one
    request instead of N slow /reset round-trips.
    """
    rng = np.random.RandomState(seed)
    gamefiles = list(_tw.game_files)
    rng.shuffle(gamefiles)
    return gamefiles


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
            _load_env()
            self._send(200, {"status": "ok", "num_games": _tw.num_games, "game_idx": _game_idx})
        elif self.path.startswith("/gamefiles"):
            _load_env()
            qs = parse_qs(urlparse(self.path).query)
            seed = int(qs.get("seed", ["1234"])[0])
            self._send(200, {"seed": seed, "num_games": _tw.num_games,
                             "gamefiles": _gamefiles_order(seed)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        global _game_idx
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")
        env = _load_env()

        if self.path.rstrip("/") == "/reset":
            game_idx = data.get("game_idx")
            seed = int(data.get("seed", 1234))
            if game_idx is not None:
                game_idx = int(game_idx)
                if not (0 <= game_idx < _tw.num_games):
                    self._send(400, {"error": f"game_idx {game_idx} outside [0, {_tw.num_games})"})
                    return
                # Re-seed to the requested order, then advance to the requested game.
                # (skip only moves forward; re-seeding resets the iterator.) The seed
                # chooses WHICH game maps to game_idx — different seeds yield
                # different held-out task sets for multi-seed evaluation.
                env.seed(seed)
                env.skip(game_idx)
            obs, infos = env.reset()
            _game_idx += 1
            self._send(200, {
                "observation": _first(obs),
                "admissible_commands": _first(infos["admissible_commands"]),
                "expert_plan": _expert_plan(infos),
                "expert_plan_full": _expert_plan_full(infos),
                "gamefile": _gamefile(infos),
                "game_idx": game_idx if game_idx is not None else _game_idx - 1,
                "done": False,
            })
        elif self.path.rstrip("/") == "/step":
            action = data.get("action", "")
            obs, _scores, dones, infos = env.step([action])
            won = infos.get("won")
            won = _first(won) if isinstance(won, (list, tuple)) else bool(won)
            self._send(200, {
                "observation": _first(obs),
                "won": bool(won),
                "done": bool(_first(dones)),
                "admissible_commands": _first(infos["admissible_commands"]),
                "expert_plan": _expert_plan(infos),
                "expert_plan_full": _expert_plan_full(infos),
                "gamefile": _gamefile(infos),
            })
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *args):  # silence request logs
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18082)
    args = ap.parse_args()
    _load_env()
    print(f"[alfworld-env] split={SPLIT} num_games={_tw.num_games} on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
