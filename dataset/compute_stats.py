"""Compute return statistics (mean, std) for collected game datasets.

Aggregates returns across all available seeds per game and saves a stats.json
in each game directory. Run this once after data collection, or again after
adding more games or seeds.

Usage:
    python dataset/compute_stats.py
    python dataset/compute_stats.py --data_dir dataset/data
"""

import argparse
import json
from pathlib import Path

import numpy as np


def compute_and_save_game_stats(game_dir: Path) -> dict | None:
    """Compute return mean/std across all seeds for a game and save stats.json."""
    seed_files = sorted(game_dir.glob('seed_*.npz'))
    # Exclude obs chunk files (seed_0_obs_000.npz etc.)
    seed_files = [f for f in seed_files if '_obs_' not in f.name]

    if not seed_files:
        return None

    all_returns = []
    all_rewards = []
    for npz_path in seed_files:
        data = np.load(npz_path)
        all_returns.append(data['returns'])
        all_rewards.append(data['rewards'])

    returns = np.concatenate(all_returns)
    rewards = np.concatenate(all_rewards)
    stats = {
        'return_mean': float(np.mean(returns)),
        'return_std': float(np.std(returns)),
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'num_seeds': len(seed_files),
        'num_steps': int(len(returns)),
    }
    with open(game_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-game return statistics for rescaling.')
    parser.add_argument('--data_dir', default='dataset/data',
                        help='Directory containing per-game data folders')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory not found: {data_dir}')

    game_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not game_dirs:
        print(f'No game directories found in {data_dir}')
        return

    for game_dir in game_dirs:
        stats = compute_and_save_game_stats(game_dir)
        if stats is None:
            print(f'{game_dir.name}: no data, skipping')
            continue
        print(f'{game_dir.name} ({stats["num_seeds"]} seeds, {stats["num_steps"]} steps): '
              f'return mean={stats["return_mean"]:.2f}, std={stats["return_std"]:.2f} | '
              f'reward mean={stats["reward_mean"]:.2f}, std={stats["reward_std"]:.2f}')

    print('Done.')


if __name__ == '__main__':
    main()
