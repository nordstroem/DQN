import gym
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import numpy as np
from collections import deque
import random
import math


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


@dataclass()
class Transition:
    state: np.ndarray
    action: float
    next_state: np.ndarray
    reward: float
    is_terminal: bool


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

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)

    def to_input(s: np.ndarray, a: float):
        x = np.concatenate([s, [a]]).astype(np.float32)
        return torch.from_numpy(x)

    memory = ReplayMemory(256)

    gamma = 0.99
    batch_size = 32
    for episode in range(500):
        epsilon = max(0.0, 1.0 - episode / 250)
        current_state = env.reset()
        iteration = 0
        total_reward = 0
        while True:
            iteration += 1
            # Epsilon greedy selection
            if np.random.rand() > epsilon:
                with torch.no_grad():
                    q_set = np.array([policy_net(to_input(current_state, ap)).item() for ap in [0.0, 1.0]])
                    print(q_set)
                next_action = np.argmax(q_set)
            else:
                next_action = np.random.randint(2)

            next_state, reward, done, _ = env.step(next_action)
            total_reward += 1
            next_transition = Transition(current_state, next_action, next_state, reward, done)
            memory.append(next_transition)
            current_state = next_state

            # Optimize q table
            if len(memory) >= batch_size:
                mini_batch = memory.random_batch(batch_size)
                targets = []
                inputs = []
                for transition in mini_batch:
                    if transition.is_terminal:
                        target = transition.reward
                    else:
                        with torch.no_grad():
                            next_q_set = [target_net(to_input(transition.next_state, ap)).item() for ap in [0.0, 1.0]]
                        target = transition.reward + gamma * max(next_q_set)

                    inputs.append(to_input(transition.state, transition.action))
                    targets.append(torch.tensor(target))

                y = torch.stack(targets)
                x = torch.stack(inputs)
                policy_net.zero_grad()
                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(policy_net(x), y)
                loss.backward()
                optimizer.step()

            env.render()

            if done:
                break

        print(total_reward)
        if episode % 1 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()


run()
