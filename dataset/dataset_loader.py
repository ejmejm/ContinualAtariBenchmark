"""Loads pre-recorded trajectory data for use in the prediction benchmark."""

from pathlib import Path
from typing import List

import numpy as np

OBS_CHUNK_SIZE = 10000


def sanitize_game_name(game_name):
    """Convert 'ALE/Pong-v5' to 'Pong'."""
    return game_name.replace('ALE/', '').replace('-v5', '')


class PrerecordedDataset:
    """Loads and serves pre-recorded trajectory data step-by-step.

    Observations are lazy-loaded one chunk at a time from compressed .npz
    files to limit memory usage. Small arrays (rewards, values, returns, etc.)
    are loaded upfront for all games.
    """

    def __init__(self, data_dir: str, game_order: List[str],
                 steps_per_game: int, seed: int = 0):
        self.game_order = game_order
        self.steps_per_game = steps_per_game
        self.total_steps = len(game_order) * steps_per_game
        self.observation_shape = (210, 160, 3)

        self._data_dir = Path(data_dir)
        self._seed = seed

        # Load small arrays for all games upfront
        rewards = []
        dones = []
        value_preds = []
        returns = []

        for game in game_order:
            game_name = sanitize_game_name(game)
            npz_path = self._data_dir / game_name / f'seed_{seed}.npz'
            data = np.load(npz_path)
            n = min(steps_per_game, len(data['rewards']))
            rewards.append(data['rewards'][:n])
            dones.append(data['dones'][:n])
            value_preds.append(data['value_predictions'][:n])
            returns.append(data['returns'][:n])

        self._rewards = np.concatenate(rewards)
        self._dones = np.concatenate(dones)
        self._value_preds = np.concatenate(value_preds)
        self._returns = np.concatenate(returns)

        # Lazy observation chunk loading state
        self._current_chunk_key = None
        self._current_chunk_obs = None

    def __len__(self):
        return self.total_steps

    def _load_obs_chunk(self, game_idx, chunk_idx):
        """Load an observation chunk from its compressed .npz file."""
        key = (game_idx, chunk_idx)
        if key == self._current_chunk_key:
            return

        game = self.game_order[game_idx]
        game_name = sanitize_game_name(game)
        chunk_path = (self._data_dir / game_name
                      / f'seed_{self._seed}_obs_{chunk_idx:03d}.npz')
        data = np.load(chunk_path)
        self._current_chunk_obs = data['observations']
        self._current_chunk_key = key

    def get_obs(self, global_step):
        game_idx = global_step // self.steps_per_game
        local_step = global_step % self.steps_per_game
        chunk_idx = local_step // OBS_CHUNK_SIZE
        idx_in_chunk = local_step % OBS_CHUNK_SIZE
        self._load_obs_chunk(game_idx, chunk_idx)
        return self._current_chunk_obs[idx_in_chunk]

    def get_reward(self, global_step):
        return float(self._rewards[global_step])

    def get_value_prediction(self, global_step):
        return float(self._value_preds[global_step])

    def get_return(self, global_step):
        return float(self._returns[global_step])

    def get_game_name(self, global_step):
        game_idx = global_step // self.steps_per_game
        return self.game_order[game_idx]

    def get_done(self, global_step):
        return bool(self._dones[global_step])
