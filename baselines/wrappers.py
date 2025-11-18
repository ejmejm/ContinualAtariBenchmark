import time
import warnings
from typing import Optional, Tuple

import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper


class VecMonitor(VecEnvWrapper):
    """
    A vectorized monitor wrapper for *vectorized* Gym environments,
    it is used to record the episode reward, length, time and other data.

    Some environments like `openai/procgen <https://github.com/openai/procgen>`_
    or `gym3 <https://github.com/openai/gym3>`_ directly initialize the
    vectorized environments, without giving us a chance to use the ``Monitor``
    wrapper. So this class simply does the job of the ``Monitor`` wrapper on
    a vectorized level.

    :param venv: The vectorized environment
    :param filename: the location to save a log file, can be None for no log
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    def __init__(
        self,
        venv: VecEnv,
        filename: Optional[str] = None,
        info_keywords: Tuple[str, ...] = (),
        log_freq: int = 100,
    ):
        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor, ResultsWriter

        # This check is not valid for special `VecEnv`
        # like the ones created by Procgen, that does follow completely
        # the `VecEnv` interface
        try:
            is_wrapped_with_monitor = venv.env_is_wrapped(Monitor)[0]
        except AttributeError:
            is_wrapped_with_monitor = False

        if is_wrapped_with_monitor:
            warnings.warn(
                "The environment is already wrapped with a `Monitor` wrapper"
                "but you are wrapping it with a `VecMonitor` wrapper, the `Monitor` statistics will be"
                "overwritten by the `VecMonitor` ones.",
                UserWarning,
            )

        VecEnvWrapper.__init__(self, venv)
        self.log_freq = log_freq
        self.step_count = 0
        self.running_reward = np.zeros(self.num_envs, dtype=np.float32)
        self.t_start = time.time()

        env_id = None
        if hasattr(venv, "spec") and venv.spec is not None:
            env_id = venv.spec.id

        self.results_writer: Optional[ResultsWriter] = None
        if filename:
            self.results_writer = ResultsWriter(
                filename, header={"t_start": self.t_start, "env_id": str(env_id)}, extra_keys=info_keywords
            )

        self.info_keywords = info_keywords
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.step_count = 0
        self.running_reward = np.zeros(self.num_envs, dtype=np.float32)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        self.running_reward += np.array(rewards)
        self.episode_lengths += 1
        self.step_count += 1
        new_infos = list(infos)
        if self.step_count % self.log_freq == 0:
            for i in range(len(infos)):
                new_infos[i] = {'episode': {
                    'r': self.running_reward[i] / self.log_freq,
                    'l': self.log_freq,
                    't': round(time.time() - self.t_start, 6),
                }}
                self.running_reward[:] = 0
                if self.results_writer:
                    self.results_writer.write_row(new_infos[i]['episode'])
        return obs, rewards, dones, new_infos

    def close(self) -> None:
        if self.results_writer:
            self.results_writer.close()
        return self.venv.close()