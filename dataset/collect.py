"""Collects trajectory data from pretrained Rainbow DQN agents playing Atari games.

Records observations, actions, rewards, value predictions, and computes accurate
discounted returns respecting episode boundaries. The final episode is always
played to completion before truncating to num_steps, ensuring return accuracy.

Observations are saved in compressed chunks to avoid OOM on large runs.

Usage:
    python dataset/collect.py
    python dataset/collect.py games="['ALE/Pong-v5']" seeds="[0]" num_steps=1000
    python dataset/collect.py num_workers=4
"""

import logging
import multiprocessing as mp
from pathlib import Path
import sys

sys.path.append('baselines/')

import ale_py
import gymnasium as gym
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from baseline_agent import PretrainedAtariAgent
from continual_atari_benchmark import ContinualAtariEnv

logger = logging.getLogger(__name__)

GAMMA = 0.99
FRAMESKIP = 4
OBS_CHUNK_SIZE = 10000


def compute_returns(rewards, dones):
    """Compute discounted returns respecting episode boundaries.

    At terminal states (dones[i] == True), the return is just the reward
    at that step — no future rewards bleed across episode boundaries.
    """
    returns = np.zeros(len(rewards), dtype=np.float64)
    returns[-1] = rewards[-1]
    for i in range(len(rewards) - 2, -1, -1):
        if dones[i]:
            returns[i] = rewards[i]
        else:
            returns[i] = rewards[i] + GAMMA * returns[i + 1]
    return returns


def sanitize_game_name(game_name):
    """Convert 'ALE/Pong-v5' to 'Pong'."""
    return game_name.replace('ALE/', '').replace('-v5', '')


def collect_game_data(game, seed, num_steps, bin_reward, output_dir,
                      progress_queue=None):
    """Play a game with the pretrained agent and save trajectory data.

    Plays at least num_steps macro-steps (each is FRAMESKIP raw frames).
    If the last episode is still running at num_steps, continues until it
    completes. Returns are computed over all collected steps, then everything
    is truncated to exactly num_steps.

    Observations are flushed to disk in compressed chunks to limit memory usage.

    If progress_queue is provided, sends step counts to it for external progress
    tracking. Otherwise displays its own tqdm bar.
    """
    gym.register_envs(ale_py)

    # Create env the same way ContinualAtariEnv does
    env = gym.make(game, full_action_space=True, frameskip=1)

    # Create a ContinualAtariEnv wrapper so PretrainedAtariAgent can initialize.
    # Point curr_game to the real env so action remapping works.
    wrapper_env = ContinualAtariEnv([game], num_steps)
    wrapper_env.curr_game = env
    behavior_agent = PretrainedAtariAgent(wrapper_env)

    # Prepare output directory for observation chunks
    game_name = sanitize_game_name(game)
    out_path = Path(output_dir) / game_name
    out_path.mkdir(parents=True, exist_ok=True)

    # Small arrays — accumulated in memory (tiny: ~few MB for 200k steps)
    actions = []
    rewards_list = []
    dones = []
    value_preds = []

    # Observation chunk — flushed to disk every OBS_CHUNK_SIZE frames
    obs_chunk = []
    chunk_idx = 0

    rng = np.random.RandomState(seed)
    obs, info = env.reset(seed=int(rng.randint(2**31)))
    terminated = False

    # Build initial info dict matching ContinualAtariEnv format
    prior_info = {
        'game_name': game,
        'reset': True,
        'skipped_frames': [],
        'terminated': False,
        'truncated': False,
    }

    step = 0
    pbar = None
    if progress_queue is None:
        pbar = tqdm(total=num_steps, desc=game_name, unit='step')

    while step < num_steps or not terminated:
        # Get action and value prediction
        action, value = behavior_agent.act_with_value(obs, prior_info)

        # Manual frameskip matching ContinualAtariEnv.step()
        skipped_frames = []
        total_reward = 0
        for _ in range(FRAMESKIP):
            next_obs, reward, term, trunc, info = env.step(action)
            skipped_frames.append(next_obs)
            total_reward += reward

        if bin_reward:
            total_reward = np.sign(total_reward)

        terminated = term or trunc

        # Record small arrays (always, even past num_steps for return computation)
        actions.append(action)
        rewards_list.append(total_reward)
        dones.append(terminated)
        value_preds.append(value)

        # Record observation only up to num_steps (save to disk in chunks)
        if step < num_steps:
            obs_chunk.append(obs)
            if len(obs_chunk) >= OBS_CHUNK_SIZE:
                chunk_path = out_path / f'seed_{seed}_obs_{chunk_idx:03d}.npz'
                np.savez_compressed(chunk_path, observations=np.array(obs_chunk, dtype=np.uint8))
                obs_chunk = []
                chunk_idx += 1

        # Prepare for next step
        prior_info = {
            'game_name': game,
            'reset': False,
            'skipped_frames': skipped_frames[:-1],
            'terminated': terminated,
            'truncated': trunc,
        }

        if terminated:
            obs, info = env.reset(seed=int(rng.randint(2**31)))
            prior_info['reset'] = True
            prior_info['skipped_frames'] = []
        else:
            obs = next_obs

        step += 1
        if pbar is not None:
            pbar.update(1)
        elif step <= num_steps:
            progress_queue.put((game, seed, 1))

    if pbar is not None:
        pbar.close()
    env.close()

    # Flush remaining observation chunk
    if obs_chunk:
        chunk_path = out_path / f'seed_{seed}_obs_{chunk_idx:03d}.npz'
        np.savez_compressed(chunk_path, observations=np.array(obs_chunk, dtype=np.uint8))

    # Convert small arrays
    actions_arr = np.array(actions, dtype=np.int32)
    rewards_arr = np.array(rewards_list, dtype=np.float32)
    dones_arr = np.array(dones, dtype=bool)
    value_preds_arr = np.array(value_preds, dtype=np.float32)

    # Compute returns over ALL collected steps (including past num_steps)
    returns = compute_returns(rewards_arr, dones_arr)

    # Truncate small arrays to num_steps
    actions_arr = actions_arr[:num_steps]
    rewards_arr = rewards_arr[:num_steps]
    dones_arr = dones_arr[:num_steps]
    value_preds_arr = value_preds_arr[:num_steps]
    returns = returns[:num_steps]

    # Save small arrays
    np.savez(
        out_path / f'seed_{seed}.npz',
        actions=actions_arr,
        rewards=rewards_arr,
        dones=dones_arr,
        value_predictions=value_preds_arr,
        returns=returns,
    )

    return (f'{game} seed={seed}: collected {step} steps '
            f'(truncated to {num_steps}), '
            f'{chunk_idx + 1} obs chunks, '
            f'avg reward={rewards_arr.mean():.3f}, '
            f'avg value={value_preds_arr.mean():.3f}, '
            f'avg return={returns.mean():.3f}')


def _worker(args):
    """Multiprocessing worker that calls collect_game_data."""
    game, seed, num_steps, bin_reward, output_dir, progress_queue = args
    progress_queue.put(('START', game, seed))
    result = collect_game_data(game, seed, num_steps, bin_reward, output_dir,
                               progress_queue)
    progress_queue.put(('DONE', game, seed))
    return result


@hydra.main(version_base=None, config_path='.', config_name='collect_config')
def main(cfg: DictConfig):
    gym.register_envs(ale_py)

    num_workers = cfg.get('num_workers', 1)
    jobs = [(game, seed) for game in cfg.games for seed in cfg.seeds]

    if num_workers <= 1:
        # Sequential — each job gets its own tqdm bar
        for game, seed in jobs:
            result = collect_game_data(game, seed, cfg.num_steps,
                                       cfg.bin_reward, cfg.output_dir)
            logger.info(result)
    else:
        # Parallel — fixed bars, one per worker slot, reused across jobs
        manager = mp.Manager()
        progress_queue = manager.Queue()

        worker_args = [
            (game, seed, cfg.num_steps, cfg.bin_reward, cfg.output_dir,
             progress_queue)
            for game, seed in jobs
        ]

        # Create one persistent bar per worker slot
        slot_bars = [tqdm(total=cfg.num_steps, desc='(idle)', unit='step',
                          position=i) for i in range(num_workers)]
        free_slots = list(range(num_workers))
        job_to_slot = {}

        pool = mp.Pool(num_workers)
        async_result = pool.map_async(_worker, worker_args)
        pool.close()

        # Drain progress queue until all workers finish
        done_count = 0
        while done_count < len(jobs):
            msg = progress_queue.get()
            if msg[0] == 'START':
                _, game, seed = msg
                slot = free_slots.pop(0)
                job_to_slot[(game, seed)] = slot
                label = f'{sanitize_game_name(game)} s{seed}'
                slot_bars[slot].reset(total=cfg.num_steps)
                slot_bars[slot].set_description(label)
            elif msg[0] == 'DONE':
                _, game, seed = msg
                slot = job_to_slot.pop((game, seed))
                slot_bars[slot].set_description('(idle)')
                free_slots.append(slot)
                done_count += 1
            else:
                game, seed, delta = msg
                slot_bars[job_to_slot[(game, seed)]].update(delta)

        for bar in slot_bars:
            bar.close()

        results = async_result.get()
        pool.join()
        print()
        for result in results:
            logger.info(result)

    logger.info('Done collecting all datasets.')


if __name__ == '__main__':
    main()
