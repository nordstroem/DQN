import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import math
from typing import NamedTuple, Any

device = torch.device("cpu")


class DQN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class Transition(NamedTuple):
    state: Any
    action: Any
    next_state: Any
    reward: Any
    is_terminal: Any


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
        batches = [self.memory[i] for i in indices]

        # Convert the list of transitions to a transition of tensors
        def to_tensor(array):
            dtype = bool if array.dtype == bool else torch.float32
            return torch.tensor(array, dtype=dtype, device=device)

        return Transition(*(to_tensor(np.array(i)) for i in zip(*batches)))


class Learner:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.num_features = 5
        self.policy_net = DQN(self.num_features).to(device)
        self.target_net = DQN(self.num_features).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory(5000)
        self.BATCH_SIZE = 32
        self.MAX_EPISODES = 5000
        self.GAMMA = 0.9999999

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.policy_net.apply(init_weights)
        self.target_net.apply(init_weights)

    def __del__(self):
        self.env.close()

    def learn(self):
        step_counter = 0
        total_rewards = []
        for episode in range(self.MAX_EPISODES):
            current_state = self.env.reset()
            iteration = 0
            total_reward = 0
            epsilon = 0.05 + (0.9 - 0.05) * math.exp(-1. * episode / 200)
            while iteration < 1000:
                iteration += 1
                next_action = self.get_action(epsilon, current_state)
                next_state, reward, done, _ = self.env.step(next_action)
                step_counter += 1
                total_reward += reward

                self.memory.append(Transition(current_state, next_action, next_state, reward, done))
                current_state = next_state
                moving_reward_average = np.mean(total_rewards[-10:]) if len(total_rewards) > 10 else 0.0

                if step_counter % 1000 == 0 and step_counter > 0:
                    print(
                        f"episode: {episode}, avg. reward: {moving_reward_average}, "
                        f"max. reward: {np.max(total_rewards[-10:])} epsilon: {epsilon}")
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                if moving_reward_average > 300:
                    self.env.render()

                # Optimize q table
                if step_counter % 3 == 0:
                    self.optimize_step()

                if done:
                    break

            total_rewards.append(total_reward)

    def get_action(self, epsilon, current_state):
        # Epsilon greedy selection
        if np.random.rand() > epsilon:
            x = torch.tensor(current_state, device=device).unsqueeze(0)
            x = x.repeat((2, 1))
            a = torch.tensor([0, 1], device=device).unsqueeze(1)
            x = torch.cat((x, a), 1)
            with torch.no_grad():
                y = self.policy_net(x)
            _, next_action = torch.max(y.view(-1), dim=0)
            next_action = next_action.item()
        else:
            next_action = np.random.randint(2)

        return next_action

    def optimize_step(self):
        if len(self.memory) >= self.BATCH_SIZE:
            transition = self.memory.random_batch(self.BATCH_SIZE)

            left_a = torch.zeros(self.BATCH_SIZE, device=device).unsqueeze(1)
            right_a = torch.ones(self.BATCH_SIZE, device=device).unsqueeze(1)

            left_x = torch.cat((transition.next_state, left_a), dim=1)
            right_x = torch.cat((transition.next_state, right_a), dim=1)

            x = torch.zeros((self.BATCH_SIZE * 2, self.num_features), device=device)
            x[::2, :] = left_x
            x[1::2, :] = right_x

            with torch.no_grad():
                y = self.target_net(x)  # Optimize with skipping terminal states?
            q_values, _ = y.view(self.BATCH_SIZE, 2).max(dim=1)

            targets = transition.reward
            targets[~transition.is_terminal] += self.GAMMA * q_values[~transition.is_terminal]

            x = torch.cat((transition.state, transition.action.view(-1, 1)), dim=1)

            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(self.policy_net(x), targets.view(-1, 1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    learner = Learner()
    learner.learn()
