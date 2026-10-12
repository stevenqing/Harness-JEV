"""Ray-backed native ScienceWorld environments."""

from typing import List, Optional

import gym
import ray


class ScienceWorldWorker:
    def __init__(self, seed: int, env_step_limit: int, server_path: Optional[str] = None):
        from scienceworld import ScienceWorldEnv
        self.env = ScienceWorldEnv(serverPath=server_path, envStepLimit=env_step_limit)
        self.task_name = None
        self.variation_idx = 0

    def reset(self, task_name: str, variation_idx: int):
        self.task_name = task_name
        self.variation_idx = int(variation_idx)
        self.env.load(self.task_name, self.variation_idx)
        observation, info = self.env.reset()
        return self._result(observation, info, 0.0, False)

    def step(self, action: str):
        observation, reward, done, info = self.env.step(action)
        return self._result(observation, info, reward, done)

    def _result(self, observation, info, reward, done):
        info = dict(info or {})
        score = float(info.get("score", 0.0))
        info.update({
            "task_name": self.task_name,
            "variation_idx": self.variation_idx,
            "task_description": info.get("taskDesc", ""),
            "goal_progress": self.env.get_goal_progress(),
            "task_score": score,
            "won": bool(score >= 100.0),
            "success": bool(score >= 100.0),
            "admissible_actions": list(self.env.get_valid_action_object_combinations()),
        })
        return str(observation), float(reward), bool(done), info

    def close(self):
        self.env.close()


class ScienceWorldEnvs(gym.Env):
    def __init__(self, seed=0, env_num=1, group_n=1, is_train=True,
                 env_config=None, resources_per_worker=None):
        super().__init__()
        if not ray.is_initialized():
            ray.init()
        self.batch_size = int(env_num * group_n)
        self.group_n = int(group_n)
        self.env_step_limit = int(env_config.max_steps)
        server_path = env_config.get("server_path", None)
        worker_cls = ray.remote(**(resources_per_worker or {}))(ScienceWorldWorker)
        self.workers = [worker_cls.remote(seed + i, self.env_step_limit, server_path)
                        for i in range(self.batch_size)]
        self.task_names = self._get_task_names(server_path)
        self._closed = False
        self._last_infos = [{} for _ in self.workers]

    def _get_task_names(self, server_path):
        from scienceworld import ScienceWorldEnv
        env = ScienceWorldEnv(serverPath=server_path, envStepLimit=self.env_step_limit)
        try:
            return list(env.get_task_names())
        finally:
            env.close()

    def _kwargs(self, kwargs, index):
        kwargs = kwargs or {}
        task_name = kwargs.get("task_name", kwargs.get("taskName"))
        if task_name is None:
            task_name = self.task_names[index % len(self.task_names)]
        return task_name, int(kwargs.get("variation_idx", kwargs.get("variationIdx", 0)))

    def reset(self, kwargs=None):
        kwargs = [] if kwargs is None else list(kwargs)
        requested = len(kwargs) or self.batch_size
        if not kwargs:
            kwargs = [{"task_index": i} for i in range(self.batch_size)]
        if len(kwargs) > self.batch_size:
            raise ValueError("More reset tasks than ScienceWorld workers")
        padded = kwargs + [{"task_index": i} for i in range(len(kwargs), self.batch_size)]
        results = ray.get([w.reset.remote(*self._kwargs(k, i))
                           for i, (w, k) in enumerate(zip(self.workers, padded))])
        observations, _, _, infos = zip(*results)
        self._last_infos = list(infos)
        self._active_count = requested
        return list(observations[:requested]), list(infos[:requested])

    def step(self, actions: List[str]):
        if len(actions) > self.batch_size:
            raise ValueError("More actions than ScienceWorld workers")
        active = getattr(self, "_active_count", self.batch_size)
        padded = list(actions) + ["look around"] * (self.batch_size - len(actions))
        results = ray.get([w.step.remote(action) for w, action in zip(self.workers, padded)])
        observations, rewards, dones, infos = zip(*results)
        self._last_infos = list(infos)
        return (list(observations[:active]), list(rewards[:active]),
                list(dones[:active]), list(infos[:active]))

    @property
    def get_admissible_commands(self):
        return [info.get("admissible_actions", []) for info in self._last_infos]

    def close(self):
        if not self._closed:
            ray.get([w.close.remote() for w in self.workers])
            self._closed = True


def build_scienceworld_envs(seed=0, env_num=1, group_n=1, is_train=True,
                             env_config=None, resources_per_worker=None):
    return ScienceWorldEnvs(seed, env_num, group_n, is_train, env_config, resources_per_worker)
