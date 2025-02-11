from typing import Any, Dict, Tuple

import numpy as np


def init(observation_shape: Tuple[int, ...]) -> Any:
    return {}


def step(state: Any, previous_observation: np.ndarray, observation: np.ndarray, reward: float) -> Tuple[Any, float, Dict[str, float]]:
    return state, 0.0, {}