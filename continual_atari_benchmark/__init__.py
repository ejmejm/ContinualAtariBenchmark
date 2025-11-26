"""Continual Atari Benchmark

A benchmark for continual learning on sequences of Atari games.
"""


__version__ = "0.1.0"


from gymnasium.envs.registration import register

from continual_atari_benchmark.env import ContinualAtariEnv


# Default game sequences for the benchmark
DEFAULT_GAME_ORDER = [
    "ALE/Pong-v5",
    "ALE/Breakout-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/Qbert-v5",
    "ALE/Seaquest-v5",
    "ALE/BeamRider-v5",
    "ALE/Enduro-v5",
    "ALE/Asterix-v5",
    "ALE/MsPacman-v5",
    "ALE/Alien-v5",
    "ALE/Freeway-v5",
    "ALE/TimePilot-v5",
    "ALE/Riverraid-v5",
    "ALE/Assault-v5",
    "ALE/RoadRunner-v5",
    "ALE/Kangaroo-v5",
    "ALE/Jamesbond-v5",
    "ALE/Krull-v5",
    "ALE/KungFuMaster-v5",
    "ALE/PrivateEye-v5",
]


def register_envs():
    """Register all continual Atari environments with Gymnasium."""
    register(
        id="ContinualAtari-v0",
        entry_point="continual_atari_benchmark.env:ContinualAtariEnv",
        kwargs={
            "game_order": DEFAULT_GAME_ORDER,
            "steps_per_game": 100_000,
            "randomize_game_order": False,
            "bin_reward": True,
        },
    )


# Register environments on import
register_envs()


__all__ = [
    "ContinualAtariEnv",
    "DEFAULT_GAME_ORDER",
    "register_envs",
]

