import gym
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import numpy as np
from collections import deque
import random


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


@dataclass()
class Transition:
    state: np.ndarray
    action: float
    next_state: np.ndarray
    reward: float


class ReplayMemory:
    def __init__(self, max_len):
        self.memory = deque(maxlen=max_len)

    def __len__(self):
        return len(self.memory)

    def append(self, transition):
        self.memory.append(transition)

    def random_batch(self, batch_size):
        assert batch_size <= len(self.memory)
        indices = random.sample(range(0, len(self.memory)), batch_size)
        return [self.memory[i] for i in indices]


def test():
    pass


test()


def run():
    env = gym.make('CartPole-v0')

    policy_network = DQN()
    target_network = DQN()
    memory = ReplayMemory(100)

    epsilon = 0.1
    for episode in range(50):
        current_state = None
        iteration = 0
        while iteration < 500:
            iteration += 1
            for i in [0, 1]:
                a = DQN()
        target_network.load_state_dict(policy_network.state_dict())


def tmp():
    env = gym.make('CartPole-v0')
    env.reset()

    for _ in range(10):
        #    env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
    env.close()
