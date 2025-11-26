import os
import sys

sys.path.append('../')

import ale_py
import gymnasium as gym
import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from continual_atari_benchmark import ContinualAtariEnv
from wrappers import VecMonitor


def make_env(cfg: DictConfig) -> gym.Env:
    env = ContinualAtariEnv(
        cfg.game_order, cfg.steps_per_game, randomize_game_order=True)
    return env


def train_policy(cfg: DictConfig) -> None:
    env = make_vec_env(lambda: make_env(cfg), n_envs=cfg.train.n_envs, vec_env_cls=SubprocVecEnv)
    env = VecMonitor(env, os.path.join('logs', 'monitor.csv'))
    batch_size = 64
    n_steps = batch_size * cfg.train.n_envs * 4
    model = PPO('MlpPolicy', env, verbose=1, batch_size=batch_size, n_steps=n_steps)
    model.learn(total_timesteps=cfg.train.total_timesteps)
    model.save(os.path.join('models', cfg.train.model_name))


@hydra.main(version_base=None, config_path='../config', config_name='train_policy')
def main(cfg: DictConfig) -> None:
    """Main entry point for running the benchmark."""
    os.makedirs('logs', exist_ok=True)
    gym.register_envs(ale_py)
    train_policy(cfg)


if __name__ == '__main__':
    main()
