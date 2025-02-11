import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Add pfrl directory to path
sys.path.append(str(Path(__file__).parent / 'pfrl'))
sys.path.append('../')

import cv2
import gymnasium as gym
import numpy as np
from pfrl import agents
from pfrl import replay_buffers, utils
from pfrl import nn as pnn
from pfrl.q_functions import DistributionalDuelingDQN

from env import ContinualAtariEnv


logger = logging.getLogger(__name__)


def download_atari_models(games: List[str]) -> List[str]:
    """Downloads and returns a list of paths to pretrained models for a list of Atari games.

    Args:
        games (List[str]): List of Atari game names.

    Returns:
        List[str]: List of paths to the downloaded models.
    """
    model_paths = []
    for game in games:
        model_paths.append(utils.download_model(
            'Rainbow', game, model_type='best')[0])
    return model_paths


def make_rainbow_agent(
        env: gym.Env,
        model_path: Optional[str] = None,
    ) -> agents.CategoricalDoubleDQN:
    """Makes a Rainbow agent and loads a pretrained model if provided."""
    n_actions = env.action_space.n

    n_atoms = 51
    v_max = 10
    v_min = -10
    q_func = DistributionalDuelingDQN(
        n_actions,
        n_atoms,
        v_min,
        v_max,
    )
    pnn.to_factorized_noisy(q_func, sigma_scale=0.5)
    
    agent = agents.CategoricalDoubleDQN(
        q_func,
        None,
        replay_buffer = replay_buffers.PrioritizedReplayBuffer(10),
        gpu = 0,
        gamma = 0.99,
        explorer = None,
        minibatch_size = 2,
        replay_start_size = 10,
        target_update_interval = 10,
        update_interval = 4,
        batch_accumulator = 'mean',
        phi = lambda x: np.asarray(x, dtype=np.float32) / 255,
    )

    agent.training = False
    
    if model_path is not None:
        agent.load(model_path)
    
    return agent


class PretrainedAtariAgent:
    """Wrapper over pretrained PFRL agents that emulates the wrappers the model was trained with."""
    
    def __init__(self, env: ContinualAtariEnv):
        self.agent = None
        self.env = env
        
        self.width = 84
        self.height = 84
        self.frame_stack = 4
        
        self.prev_frames = []
        self.game_name = None
        self.minimal_action_set = None
        self.action_set = None
        self.skipped_frames = None
        
        # Download all of the pretrained models
        logger.info('Downloading or finding cached pretrained models...')
        self.unique_games = [self._env_name_to_model_name(x) for x in set(env.game_order)]
        self.model_paths = download_atari_models(self.unique_games)
        self.model_paths = {
            game: model_path for game, model_path in
            zip(self.unique_games, self.model_paths)
        }
        logger.info('Done!')
        
    def _env_name_to_model_name(self, env_name: str) -> str:
        # Original models were trained with NoFrameskip-v4 versions, but I have made corresponding
        # Changes to the v5 environnments to make them compatible.
        return env_name.replace('ALE/', '').replace('-v5', 'NoFrameskip-v4')
    
    def get_action_remapping(self, original_action_set: List[int], new_action_set: List[int]) -> List[int]:
        return [new_action_set.index(action) if action in new_action_set else -1 for action in original_action_set]
    
    def _process_game_change(self):
        curr_game = self.env.get_current_game()
        self.minimal_action_set = curr_game.unwrapped.ale.getMinimalActionSet()
        self.action_set = curr_game.unwrapped._action_set
        self.action_remapping = self.get_action_remapping(self.minimal_action_set, self.action_set)
        self.prev_frames = []
        
        # Load the pretrained model
        logger.info(f'Loading pretrained model for {self.game_name}')
        self.agent = make_rainbow_agent(
            self.env,
            self.model_paths[self._env_name_to_model_name(self.game_name)],
        )
        
    def _process_game_reset(self):
        self.prev_frames = []
    
    def _preprocess_observation(self, obs: np.ndarray, prior_info: Dict) -> np.ndarray:
        if 'skipped_frames' in prior_info and len(prior_info['skipped_frames']) > 0:
            obs = np.stack(prior_info['skipped_frames'][-1], obs).max(axis=0) # Max pool over prior and current frame
        
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(
            obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        obs = obs.reshape((1, self.height, self.width))
        
        if len(self.prev_frames) == 0:
            self.prev_frames = [obs for _ in range(self.frame_stack)]
        
        self.prev_frames.append(obs)
        self.prev_frames = self.prev_frames[-self.frame_stack:]
        obs = np.concatenate(self.prev_frames, axis=0)
        
        return obs
    
    def act(self, obs: np.ndarray, prior_info: Dict) -> int:
        if self.game_name != prior_info['game_name']:
            self.game_name = prior_info['game_name']
            self._process_game_change()
        
        if prior_info['reset']:
            self._process_game_reset()
        
        obs = self._preprocess_observation(obs, prior_info)
        action = self.agent.act(obs)
        action = self.action_remapping[action]
        return action
