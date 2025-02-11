from pathlib import Path
import sys
import time
from typing import Tuple, List, Dict

sys.path.append('baselines/')

import gymnasium as gym
import ale_py
import hydra
import importlib
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
import wandb

from env import ContinualAtariEnv
from baseline_agent import PretrainedAtariAgent


def setup_wandb(cfg: DictConfig) -> None:
    """Sets up Weights & Biases logging if enabled in config."""
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=dict(cfg),
        )


def log_metrics(metrics: Dict[str, float], step: int, cfg: DictConfig) -> None:
    """Logs metrics to wandb if enabled."""
    print(f"Step {step}: {metrics}")
    if cfg.use_wandb:
        wandb.log(metrics, step=step)


def run_benchmark(cfg: DictConfig) -> None:
    """Runs the continual learning benchmark based on config settings."""
    
    # Import correct module based on benchmark type
    module = importlib.import_module(cfg.benchmark_type)
    init_fn = module.init
    step_fn = module.step
    
    # Track metrics
    total_reward = 0
    total_mse = 0
    step_times: List[float] = []
    metrics_history: List[Tuple[int, float]] = []
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    setup_wandb(cfg)
    
    total_steps = len(cfg.game_order) * cfg.steps_per_game
    
    # Create continual learning environment
    env = ContinualAtariEnv(cfg.game_order, cfg.steps_per_game)
    
    # Initialize
    if cfg.benchmark_type == 'control':
        state = init_fn(env.observation_space.shape, env.action_space.n)
    else:
        state = init_fn(env.observation_space.shape)
        behavior_agent = PretrainedAtariAgent(env)
            
    obs, info = env.reset()
    reward = 0
    prev_obs = obs
    
    # Run episodes
    for step in range(1, total_steps + 1):
        # Track step time
        start_time = time.time()
        
        if cfg.benchmark_type == 'control':
            state, action = step_fn(state, prev_obs, obs, 0.0)    # Initial reward=0
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            metric = total_reward / (step + 1)
            metrics_history.append((step, metric))
            
        else: # Prediction
            state, value_pred = step_fn(state, prev_obs, obs, reward)
            action = behavior_agent.act(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # For now use dummy MSE since we don't have true values
            mse = (value_pred - 0.0) ** 2    # Compare to dummy value of 0
            total_mse += mse
            metric = total_mse / (step + 1)
            metrics_history.append((step, metric))
        
        # Track performance
        step_time = time.time() - start_time
        step_times.append(step_time)
        
        if step % 1000 == 0:
            current_game = cfg.game_order[step // cfg.steps_per_game]
            metrics = {
                'step': step,
                'game': current_game,
                'avg_step_time': np.mean(step_times[-1000:]),
            }
            if cfg.benchmark_type == 'control':
                metrics['avg_reward'] = metric
            else:
                metrics['avg_mse'] = metric
            log_metrics(metrics, step, cfg)
        
        prev_obs = obs
        obs = next_obs
    
    env.close()
    
    # Save final results
    final_metrics = {
        'total_steps': total_steps,
        'avg_step_time': np.mean(step_times),
    }
    
    if cfg.benchmark_type == 'control':
        final_metrics['final_avg_reward'] = total_reward / total_steps
    else:
        final_metrics['final_avg_mse'] = total_mse / total_steps
    
    # Save metrics to file
    with open(results_dir / 'metrics.txt', 'w') as f:
        for key, value in final_metrics.items():
            f.write(f"{key}: {value}\n")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    steps, metrics = zip(*metrics_history)
    plt.plot(steps, metrics)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward' if cfg.benchmark_type == 'control' else 'Average MSE')
    plt.title(f'Learning Curve - {cfg.benchmark_type.capitalize()} Benchmark')
    plt.savefig(results_dir / 'learning_curve.png')
    plt.close()
    
    if cfg.use_wandb:
        wandb.log(final_metrics)
        wandb.finish()


@hydra.main(version_base=None, config_path="config", config_name="benchmark_config")
def main(cfg: DictConfig) -> None:
    """Main entry point for running the benchmark."""
    gym.register_envs(ale_py)
    run_benchmark(cfg)


if __name__ == '__main__':
    main()
