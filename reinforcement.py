from dataclasses import dataclass
from time import perf_counter

import gymnasium as gym
import numpy as np


def make_environment():
    return gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)


def update_q(q_table, state, action, reward, next_state, terminated):
    target = reward if terminated else reward + .95 * np.max(q_table[next_state])
    q_table[state, action] += .1 * (target - q_table[state, action])


@dataclass
class AgentResult:
    q_table: np.ndarray
    rewards: np.ndarray
    rolling_success: np.ndarray
    success_rate: float
    path: list
    status: str
    seconds: float


def train_agent():
    start = perf_counter()
    rng = np.random.default_rng(42)
    q_table = np.zeros((16, 4))
    rewards = np.zeros(10000)
    environment = make_environment()
    environment.reset(seed=42)
    epsilon = 1.0
    try:
        for episode in range(10000):
            state, _ = environment.reset()
            for _ in range(100):
                action = int(rng.integers(4)) if rng.random() < epsilon else int(np.argmax(q_table[state]))
                next_state, reward, terminated, truncated, _ = environment.step(action)
                update_q(q_table, state, action, reward, next_state, terminated)
                state = next_state
                rewards[episode] += reward
                if terminated or truncated:
                    break
            epsilon = max(.05, epsilon * .999)
    finally:
        environment.close()
    rolling = np.convolve(rewards, np.ones(100) / 100, mode='valid')
    success_rate = evaluate_policy(q_table)
    path, status = rollout(q_table)
    return AgentResult(q_table, rewards, rolling, success_rate, path, status, perf_counter() - start)


def evaluate_policy(q_table):
    environment = make_environment()
    wins = 0
    try:
        environment.reset(seed=42)
        for _ in range(100):
            state, _ = environment.reset()
            for _ in range(100):
                state, reward, terminated, truncated, _ = environment.step(int(np.argmax(q_table[state])))
                wins += int(reward)
                if terminated or truncated:
                    break
    finally:
        environment.close()
    return wins / 100


def rollout(q_table):
    environment = make_environment()
    state, _ = environment.reset(seed=42)
    path = [int(state)]
    try:
        for _ in range(100):
            state, reward, terminated, truncated, _ = environment.step(int(np.argmax(q_table[state])))
            path.append(int(state))
            if terminated:
                return path, 'goal' if reward else 'hole'
            if truncated:
                return path, 'step limit'
            if state in path[:-1]:
                return path, 'loop'
        return path, 'step limit'
    finally:
        environment.close()
