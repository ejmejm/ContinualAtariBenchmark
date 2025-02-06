# Continual Atari Benchmark

This repository contains the code for two benchmarks: the Continual Atari Control Benchmark, and the Continual Atari Prediction Benchmark.
In both benchmarks, a single agent interacts with an Atari game, and generates a tuple of the form (previous observation, observation, reward) at each timestep.
Every 10,000 timesteps, the game is changed to a new Atari game based on a predetermined sequence of games.
This switch happens 20 times, after which the benchmark ends.

### Control Benchmark

For the control benchmark, you must implement the following functions:

```python
def init(observation_shape: Tuple[int, ...], num_actions: int) -> Any:
    pass

def step(state: Any, previous_observation: np.ndarray, observation: np.ndarray, reward: float) -> Tuple[Any, int]:
    pass
```

The `init` function should initialize anything your algorithm needs (like model weights).
The value it returns will be passed to the `step` function every timestep as the first argument.

The `step` function should return a tuple of the form `(state, action)`.
The `state` will be passed to the `step` function the next timestep as the first argument.
The `action` will be the action taken by the agent at this timestep.

The performance on the control benchmark is evaluated by the cumulative reward over the course of the benchmark.

### Prediction Benchmark

For the prediction benchmark, you must implement the following functions:

```python
def init(observation_shape: Tuple[int, ...]) -> Any:
    pass

def step(state: Any, previous_observation: np.ndarray, observation: np.ndarray, reward: float) -> Tuple[Any, float]:
    pass
```

The `init` function should initialize anything your algorithm needs (like model weights).
The value it returns will be passed to the `step` function every timestep as the first argument.

The `step` function should return a tuple of the form `(state, prediction)`.
The `state` will be passed to the `step` function the next timestep as the first argument.
The `prediction` will be the prediction of the value for the current observation and pre-trained, fixed policy that provides the actions (which are not known to the algorithm) in this benchmark.

The performance on the prediction benchmark is evaluated by the mean squared error of the predicted value and our own value function.
Our value function has no time limit and is trained to convergence, so it serves as an upper bound for what is possible.
