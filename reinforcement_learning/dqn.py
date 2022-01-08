import gym
import torch
import torch.nn as nn
from dataclasses import dataclass
import numpy as np
from collections import deque
import random
import math


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
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


class Learner:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.policy_net = DQN()
        self.target_net = DQN()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory(5000)

    def __del__(self):
        self.env.close()

    def learn(self):
        def to_input(s: np.ndarray, a: float):
            x = np.concatenate([s, [a]]).astype(np.float32)
            return torch.from_numpy(x)

        gamma = 0.999999
        batch_size = 64
        step_counter = 0
        total_rewards = []
        for episode in range(5000):
            current_state = self.env.reset()
            iteration = 0
            total_reward = 0
            epsilon = 0.08 + (0.9 - 0.08) * math.exp(-1. * episode / 200)
            while iteration < 500:
                iteration += 1
                # Epsilon greedy selection
                if np.random.rand() > epsilon:
                    with torch.no_grad():
                        q_set = np.array([self.policy_net(to_input(current_state, ap)).item() for ap in [0.0, 1.0]])
                    next_action = np.argmax(q_set)
                else:
                    next_action = np.random.randint(2)

                next_state, reward, done, _ = self.env.step(next_action)
                step_counter += 1
                total_reward += 1
                next_transition = Transition(current_state, next_action, next_state, reward, done)
                self.memory.append(next_transition)
                current_state = next_state
                moving_reward_average = np.mean(total_rewards[:-10]) if len(total_rewards) > 10 else 0.0

                if step_counter % 1000 == 0 and step_counter > 0:
                    print(f"episode: {episode}, avg. reward: {moving_reward_average}, epsilon: {epsilon}")
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    # Reinitializing the optimizer here seems to give slightly better convergence
                    optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)

                if moving_reward_average > 140:
                    self.env.render()

                # Optimize q table
                if step_counter % 4 == 0:
                    if len(self.memory) >= batch_size:
                        mini_batch = self.memory.random_batch(batch_size)
                        targets = []
                        inputs = []
                        for transition in mini_batch:
                            if transition.is_terminal:
                                target = transition.reward
                            else:
                                with torch.no_grad():
                                    next_q_set = [self.target_net(to_input(transition.next_state, ap)).item() for ap in
                                                  [0.0, 1.0]]
                                target = transition.reward + gamma * max(next_q_set)

                            inputs.append(to_input(transition.state, transition.action))
                            targets.append(torch.tensor(target))

                        y = torch.stack(targets)
                        x = torch.stack(inputs)
                        self.policy_net.zero_grad()
                        loss_fn = torch.nn.MSELoss()
                        loss = loss_fn(self.policy_net(x), y.unsqueeze(1))
                        loss.backward()
                        self.optimizer.step()

                if done:
                    break
            total_rewards.append(total_reward)


if __name__ == "__main__":
    learner = Learner()
    learner.learn()
