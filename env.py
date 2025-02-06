from typing import Dict, List, Tuple, Union

import gymnasium as gym
import numpy as np


class ContinualAtariEnv(gym.Env):
    def __init__(self, game_sequence: List[str], steps_per_game: int):
        super().__init__()
        
        self.game_sequence = game_sequence
        self.steps_per_game = steps_per_game
        self.current_game_idx = 0
        self.current_step = 0
        self.seed = None
        
        self.action_space = gym.spaces.Discrete(18)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.float32)
        
        self.env = None
        self.terminated = False
        
    def reset(self, seed: Union[int, None] = None) -> Tuple[np.ndarray, Dict]:
        if self.env is not None:
            self.env.close()
            
        self.current_game_idx = 0
        self.current_step = 0
        self.terminated = False
        self.seed = seed
        
        self.env = gym.make(self.game_sequence[self.current_game_idx], full_action_space=True)
        print(self.env.action_space)
        obs, info = self.env.reset(seed=seed)
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.current_step >= self.steps_per_game:
            self.current_game_idx = (self.current_game_idx + 1) % len(self.game_sequence)
            self.current_step = 0
            
            if self.env is not None:
                self.env.close()
                
            self.env = gym.make(self.game_sequence[self.current_game_idx], full_action_space=True)
            print(self.env.action_space)
            self.seed = self.seed + 1 if self.seed is not None else None
            obs, info = self.env.reset(seed=self.seed)
            reward = 0
            self.terminated = False
        
        elif self.terminated:
            self.seed = self.seed + 1 if self.seed is not None else None
            obs, info = self.env.reset(seed=self.seed)
            reward = 0
            self.terminated = False
                
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.terminated = terminated or truncated
        
        self.current_step += 1
        
        return obs, reward, False, False, info