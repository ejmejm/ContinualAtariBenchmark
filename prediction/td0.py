from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


DEVICE = 'cuda'


def td0_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    observation: torch.Tensor,
    cumulant: torch.Tensor,
    next_observation: torch.Tensor,
    gamma: float,
    is_terminal: bool = False,
    grad_clip: Optional[float] = 2.0,
) -> float:
    """Performs a single TD(0) update step for value prediction.

    Args:
        model: Neural network that predicts values
        observation: Current observation tensor
        cumulant: Immediate reward tensor
        next_observation: Next observation tensor
        gamma: Discount factor for future rewards
        is_terminal: Whether this is a terminal state (no future rewards)
        grad_clip: Gradient clipping threshold
    
    Returns:
        float: The TD error (loss) for this update
    """
    # Get current value prediction
    prediction = model(observation)[0]
    
    # Calculate target (reward + gamma * next_value)
    with torch.no_grad():
        next_value = model(next_observation)
        target = cumulant + (0 if is_terminal else gamma * next_value)
    
    # Compute TD error
    loss = F.mse_loss(prediction, target)
    
    # Backward pass and update
    optimizer.zero_grad()
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    
    return loss.item()


@dataclass
class PredictorState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    gamma: float
    obs_decay: float
    obs_avg: np.ndarray
    obs_sq_avg: np.ndarray
    step: int


def init(observation_shape: Tuple[int, ...]) -> PredictorState:
    model = nn.Sequential(
        nn.Linear(np.prod(observation_shape), 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    ).to(DEVICE)
    model[-1].weight.data.fill_(0.0)
    model[-1].bias.data.fill_(0.0)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    return PredictorState(
        model = model,
        optimizer = optimizer,
        gamma = 0.9,
        obs_decay = 0.99,
        obs_avg = np.zeros(observation_shape),
        obs_sq_avg = np.zeros(observation_shape),
        step = 0,
    )


def step(
    state: PredictorState,
    previous_observation: np.ndarray,
    observation: np.ndarray,
    reward: float,
) -> Tuple[PredictorState, float, Dict[str, float]]:
    
    # Update running average of observations
    state.obs_avg = state.obs_decay * state.obs_avg + (1 - state.obs_decay) * observation
    state.obs_sq_avg = state.obs_decay * state.obs_sq_avg + (1 - state.obs_decay) * observation ** 2
    
    # Normalize observation
    corrected_avg = state.obs_avg / (1 - state.obs_decay ** (state.step + 1))
    corrected_sq_avg = state.obs_sq_avg / (1 - state.obs_decay ** (state.step + 1))
    norm_prev_obs = (previous_observation - corrected_avg) / (np.sqrt(corrected_sq_avg) + 1e-6)
    norm_obs = (observation - corrected_avg) / (np.sqrt(corrected_sq_avg) + 1e-6)
    
    # Format observations for model
    previous_observation = torch.from_numpy(norm_prev_obs).reshape(1, -1).float().to(DEVICE)
    observation = torch.from_numpy(norm_obs).reshape(1, -1).float().to(DEVICE)
    
    # Perform TD(0) update
    loss = td0_update(
        state.model,
        state.optimizer,
        previous_observation,
        reward,
        observation,
        state.gamma,
    )
    value_pred = state.model(observation)[0].item()
    
    # Update step counter
    state.step += 1
    
    return state, value_pred, {'td_loss': loss}