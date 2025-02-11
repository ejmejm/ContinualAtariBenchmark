from typing import Any, Dict, Tuple

import numpy as np


def init(observation_shape: Tuple[int, ...], num_actions: int) -> Any:
    return {'num_actions': num_actions}


# Random action agent example
def step(state: Any, previous_observation: np.ndarray, observation: np.ndarray, reward: float) -> Tuple[Any, int, Dict[str, float]]:
    return state, np.random.randint(state['num_actions']), {}