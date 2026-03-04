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

from continual_atari_benchmark import ContinualAtariEnv
from baseline_agent import PretrainedAtariAgent
from dataset.dataset_loader import PrerecordedDataset

GAMMA = 0.99


def compute_returns(rewards, dones):
    """Compute discounted returns respecting episode boundaries."""
    returns = np.zeros(len(rewards), dtype=np.float64)
    returns[-1] = rewards[-1]
    for i in range(len(rewards) - 2, -1, -1):
        if dones[i]:
            returns[i] = rewards[i]
        else:
            returns[i] = rewards[i] + GAMMA * returns[i + 1]
    return returns


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
    module = importlib.import_module(f'{cfg.benchmark_type}.{cfg.method}')
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
    
    use_prerecorded = (cfg.benchmark_type == 'prediction'
                       and cfg.get('use_prerecorded', False))

    total_steps = len(cfg.game_order) * cfg.steps_per_game

    if use_prerecorded:
        # Load pre-recorded dataset instead of live environment
        dataset = PrerecordedDataset(
            cfg.prerecorded_data_dir,
            cfg.game_order,
            cfg.steps_per_game,
            seed=cfg.prerecorded_seed,
        )
        state = init_fn(dataset.observation_shape)
        prev_obs = dataset.get_obs(0)
        reward = 0.0
        env = None
    else:
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

    # Setup video recording if enabled (live mode only)
    video_writer: Optional[cv2.VideoWriter] = None
    if cfg.save_video and env is not None:
        video_path = str(run_dir / 'benchmark.mp4')
        video_writer = setup_video_writer(env.render().shape, fps=15, path=video_path)

    running_metrics = defaultdict(float)
    done_flags: List[bool] = []

    # Run episodes
    for step in range(1, total_steps + 1):
        # Track step time
        start_time = time.time()

        # Save video frame if enabled
        if cfg.save_video and env is not None:
            frame = env.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        if use_prerecorded:
            i = step - 1
            obs = dataset.get_obs(i)
            state, value_pred, custom_metrics = step_fn(state, prev_obs, obs, reward)

            reward = dataset.get_reward(i)
            baseline_value = dataset.get_value_prediction(i)
            actual_return = dataset.get_return(i)

            baseline_mse = (value_pred - baseline_value) ** 2
            return_mse = (value_pred - actual_return) ** 2

            metrics = {
                'baseline_mse': baseline_mse,
                'return_mse': return_mse,
                'value_prediction': value_pred,
                '_reward': reward,
                **custom_metrics,
            }
            metrics_history.append((step, metrics))
            prev_obs = obs

        elif cfg.benchmark_type == 'control':
            state, action, custom_metrics = step_fn(state, prev_obs, obs, 0.0)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            metrics = {
                'reward': reward,
                **custom_metrics,
            }
            metrics_history.append((step, metrics))
            prev_obs = obs
            obs = next_obs

        else:  # Live prediction
            state, value_pred, custom_metrics = step_fn(state, prev_obs, obs, reward)
            action = behavior_agent.act(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            mse = (value_pred - 0.0) ** 2
            total_mse += mse
            metrics = {
                'baseline_mse': mse,
                'value_prediction': value_pred,
                '_reward': reward,
                **custom_metrics,
            }
            metrics_history.append((step, metrics))
            done_flags.append(terminated or truncated)
            prev_obs = obs
            obs = next_obs

        for key, value in metrics.items():
            running_metrics[key] += value

        # Track performance
        step_time = time.time() - start_time
        step_times.append(step_time)

        if step % cfg.log_freq == 0:
            if use_prerecorded:
                current_game = dataset.get_game_name(step - 1)
            else:
                current_game = env.get_current_game().spec.id
            log_data = {
                'step': step,
                'game': current_game,
                'avg_step_time': np.mean(step_times[-cfg.log_freq:]),
            }
            for key, value in running_metrics.items():
                if not key.startswith('_'):
                    log_data[key] = value / cfg.log_freq
            running_metrics.clear()
            log_metrics(log_data, step, cfg)

    # Cleanup
    if video_writer is not None:
        video_writer.release()
    if env is not None:
        env.close()

    # Compute returns for live prediction mode (prerecorded already has them)
    if cfg.benchmark_type == 'prediction' and not use_prerecorded:
        value_preds = np.array([m[1]['value_prediction'] for m in metrics_history])
        rewards = np.array([m[1]['_reward'] for m in metrics_history])
        dones = np.array(done_flags, dtype=bool)
        returns = compute_returns(rewards, dones)

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
        y_values = [metrics_dict[metric_name] for metrics_dict in metrics]
        plt.plot(steps, y_values)
        plt.xlabel('Steps')
        plt.ylabel(formatted_metric_name)
        plt.title(f'Learning Curve - {formatted_metric_name} Benchmark')
        
        y_top = np.percentile(y_values, 95)
        y_bottom = np.percentile(y_values, 5)
        y_range = y_top - y_bottom
        y_top = y_top + y_range * 0.1
        y_bottom = y_bottom - y_range * 0.1
        plt.ylim(y_bottom, y_top)
        
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
