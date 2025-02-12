from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Tuple, List, Dict, Optional

sys.path.append('baselines/')

import gymnasium as gym
import ale_py
import hydra
import importlib
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
import wandb
import cv2

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


def setup_video_writer(shape: Tuple[int, ...], fps: int, path: str) -> cv2.VideoWriter:
    """Sets up a VideoWriter object for saving gameplay footage.
    
    Args:
        shape: Shape of the frames (height, width, channels)
        fps: Frames per second for the output video
        path: Path where the video will be saved
    
    Returns:
        VideoWriter object configured for MP4 output
    """
    height, width = shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


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
    metrics_history: List[Tuple[int, Dict[str, float]]] = []
    
    # Create results directory with datetime subfolder
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = results_dir / timestamp
    run_dir.mkdir(exist_ok=True)
    
    setup_wandb(cfg)
    
    total_steps = len(cfg.game_order) * cfg.steps_per_game
    
    # Create continual learning environment
    env = ContinualAtariEnv(
        cfg.game_order,
        cfg.steps_per_game,
        render_mode = 'rgb_array' if cfg.save_video else None,
    )
    
    # Initialize
    if cfg.benchmark_type == 'control':
        state = init_fn(env.observation_space.shape, env.action_space.n)
    else:
        state = init_fn(env.observation_space.shape)
        behavior_agent = PretrainedAtariAgent(env)
            
    obs, info = env.reset()
    reward = 0
    prev_obs = obs
    
    # Setup video recording if enabled
    video_writer: Optional[cv2.VideoWriter] = None
    if cfg.save_video:
        video_path = str(run_dir / 'benchmark.mp4')
        video_writer = setup_video_writer(env.render().shape, fps=15, path=video_path)
    
    running_metrics = defaultdict(float)
    
    # Run episodes
    for step in range(1, total_steps + 1):
        # Track step time
        start_time = time.time()
        
        # Save video frame if enabled
        if cfg.save_video:
            frame = env.render()
            # OpenCV expects BGR format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        if cfg.benchmark_type == 'control':
            state, action, custom_metrics = step_fn(state, prev_obs, obs, 0.0)    # Initial reward=0
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            metrics = {
                # 'avg_reward': total_reward / (step + 1),
                'reward': reward,
                **custom_metrics,
            }
            metrics_history.append((step, metrics))
        
        else: # Prediction
            state, value_pred, custom_metrics = step_fn(state, prev_obs, obs, reward)
            action = behavior_agent.act(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # For now use dummy MSE since we don't have true values
            mse = (value_pred - 0.0) ** 2    # Compare to dummy value of 0
            total_mse += mse
            metrics = {
                'baseline_mse': mse,
                'value_prediction': value_pred,
                '_reward': reward,
                **custom_metrics,
            }
            metrics_history.append((step, metrics))
        
        for key, value in metrics.items():
            running_metrics[key] += value
        
        # Track performance
        step_time = time.time() - start_time
        step_times.append(step_time)
        
        if step % cfg.log_freq == 0:
            current_game = env.get_current_game().spec.id
            metrics = {
                'step': step,
                'game': current_game,
                'avg_step_time': np.mean(step_times[-cfg.log_freq:]),
            }
            for key, value in running_metrics.items():
                if not key.startswith('_'):
                    metrics[key] = value / cfg.log_freq
            running_metrics.clear()
            log_metrics(metrics, step, cfg)
        
        prev_obs = obs
        obs = next_obs
    
    # Cleanup video writer if it exists
    if video_writer is not None:
        video_writer.release()
    
    env.close()
    
    # Add difference between value predictions and discounted returns to metrics
    if cfg.benchmark_type == 'prediction':
        value_preds = np.array([metrics[1]['value_prediction'] for metrics in metrics_history])
        rewards = np.array([metrics[1]['_reward'] for metrics in metrics_history])
        returns = rewards.copy()
        for i in range(len(returns) - 2, -1, -1):
            returns[i] += cfg.discount_factor * returns[i + 1]
        
        mse = (value_preds - returns) ** 2
        for i, (step, metrics) in enumerate(metrics_history):
            metrics['return'] = returns[i]
            metrics['return_mse'] = mse[i]
    
    # Save final results
    final_metrics = {
        'total_steps': total_steps,
        'avg_step_time': np.mean(step_times),
    }
    
    # Create stats for each metric that is stored every step
    metric_keys = metrics_history[0][1].keys()
    
    for key in metric_keys:
        if not key.startswith('_'):
            final_metrics[f'final_avg_{key}'] = np.mean([metrics[1][key] for metrics in metrics_history])
    
    # Save metrics to file
    with open(run_dir / 'metrics.txt', 'w') as f:
        for key, value in final_metrics.items():
            f.write(f"{key}: {value}\n")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    steps, metrics = zip(*metrics_history)
    for metric_name in metric_keys:
        if metric_name.startswith('_'):
            continue
        formatted_metric_name = metric_name.replace('_', ' ').capitalize()
        plt.plot(steps, [metrics_dict[metric_name] for metrics_dict in metrics])
        plt.xlabel('Steps')
        plt.ylabel(formatted_metric_name)
        plt.title(f'Learning Curve - {formatted_metric_name} Benchmark')
        plt.savefig(run_dir / f'{metric_name}.png')
        plt.close()
    
    if cfg.use_wandb:
        for key, value in final_metrics.items():
            wandb.run.summary[key] = value
        wandb.finish()


@hydra.main(version_base=None, config_path='config', config_name='benchmark')
def main(cfg: DictConfig) -> None:
    """Main entry point for running the benchmark."""
    gym.register_envs(ale_py)
    run_benchmark(cfg)


if __name__ == '__main__':
    main()
