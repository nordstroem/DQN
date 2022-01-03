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


def run():
    env = gym.make('CartPole-v0')
    policy_net = DQN()
    target_net = DQN()

    def to_input(s: np.ndarray, a: np.float32):
        x = np.concatenate((current_state, [0.0])).astype(np.float32)
        return torch.from_numpy(x)

    memory = ReplayMemory(100)

    epsilon = 0.1
    gamma = 0.3
    batch_size = 8
    for episode in range(1):
        current_state = env.reset()
        iteration = 0
        while iteration < 500:
            iteration += 1
            # Epsilon greedy selection
            if np.random.rand() > epsilon:
                with torch.no_grad():
                    q_set = np.array([policy_net(to_input(current_state, ap)).item() for ap in [0, 1]])
                next_action = np.argmax(q_set)
            else:
                next_action = np.random.randint(2)

            next_state, reward, done, _ = env.step(next_action)
            next_transition = Transition(current_state, next_action, reward, next_state)
            memory.append(next_transition)
            current_state = next_state

            # Optimize q table
            if len(memory) >= batch_size:
                # loss = (torch.tensor(target) - policy_net(to_input(current_state, next_action))).pow(2).sum()
                # policy_net.zero_grad()
                # loss.backward()
                pass

            # Incorrect, needs to sample memory
            # next_q_set = [target_net(to_input(next_state, ap)).item() for ap in [0, 1]]
            # target = reward if done else reward + gamma * max(next_q_set)

            env.render()

            if done:
                break

        target_net.load_state_dict(policy_net.state_dict())

    env.close()


# run()

def tmp():
    env = gym.make('CartPole-v0')
    env.reset()

    for _ in range(10):
        #    env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
    env.close()
