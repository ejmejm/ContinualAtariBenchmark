from typing import Dict, List, Tuple, Union

import gymnasium as gym
import numpy as np


class ContinualAtariEnv(gym.Env):
    def __init__(
        self,
        game_order: List[str],
        steps_per_game: int,
        randomize_game_order: bool = False,
    ):
        super().__init__()
        
        self.game_order = game_order
        self.steps_per_game = steps_per_game
        self.randomize_game_order = randomize_game_order
        self.frameskip = 4 # Manually frameskip so that I have access to skipped frames
        self.current_game_idx = 0
        self.current_step = 0
        self.seed = None
        
        self.action_space = gym.spaces.Discrete(18)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.float32)
        
        self.curr_game = None
        self.terminated = False
        
    def reset(self, seed: Union[int, None] = None) -> Tuple[np.ndarray, Dict]:
        if self.curr_game is not None:
            self.curr_game.close()
            
        self.current_game_idx = 0
        self.current_step = 0
        self.terminated = False
        self.seed = seed
        
        if self.randomize_game_order:
            rng = np.random.default_rng(seed)
            new_order = rng.permutation(len(self.game_order)).tolist()
            self.game_order = [self.game_order[i] for i in new_order]
        
        self.curr_game = gym.make(
            self.game_order[self.current_game_idx],
            full_action_space = True,
            frameskip = 0,
        )
        obs, info = self.curr_game.reset(seed=seed)
        
        info['terminated'] = False
        info['truncated'] = False
        info['skipped_frames'] = []
        info['game_name'] = self.game_order[self.current_game_idx]
        info['reset'] = True
        
        return obs, info
    
    def get_current_game(self) -> gym.Env:
        return self.curr_game
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.current_step >= self.steps_per_game:
            self.current_game_idx = (self.current_game_idx + 1) % len(self.game_order)
            self.current_step = 0
            
            if self.curr_game is not None:
                self.curr_game.close()
                
            self.curr_game = gym.make(
                self.game_order[self.current_game_idx],
                full_action_space = True,
                frameskip = 0,
            )
            self.seed = self.seed + 1 if self.seed is not None else None
            obs, info = self.curr_game.reset(seed=self.seed)
            reward = 0
            self.terminated = False
            reset = True
        
        elif self.terminated:
            self.seed = self.seed + 1 if self.seed is not None else None
            obs, info = self.curr_game.reset(seed=self.seed)
            reward = 0
            self.terminated = False
            reset = True
            
        else:
            reset = False
        
        skipped_frames = []
        total_reward = 0
        for _ in range(self.frameskip):
            obs, reward, terminated, truncated, info = self.curr_game.step(action)
            skipped_frames.append(obs)
            total_reward += reward
        
        self.terminated = terminated or truncated
        info['terminated'] = self.terminated
        info['truncated'] = truncated
        info['skipped_frames'] = skipped_frames[:-1]
        info['game_name'] = self.game_order[self.current_game_idx]
        info['reset'] = reset

        self.current_step += 1
        
        return obs, total_reward, False, False, info