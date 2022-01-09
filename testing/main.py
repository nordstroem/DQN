import torch
import dataclasses
from dataclasses import dataclass


def optimize_step_input():
    batches = 5
    features = 3
    x = torch.arange(batches * features).reshape(batches, features)

    left_a = torch.zeros(batches).unsqueeze(1)
    right_a = torch.ones(batches).unsqueeze(1)

    left_x = torch.cat((x, left_a), 1)
    right_x = torch.cat((x, right_a), 1)

    x = torch.zeros((batches * 2, features + 1))
    x[::2, :] = left_x
    x[1::2, :] = right_x

    y = torch.arange(batches * 2).reshape(batches * 2, 1)
    res = y.view(batches, 2)

    values, indices = res.max(dim=1)

    print(values, indices)


# optimize_step_input()

@dataclass()
class Transition:
    state: float
    action: float


t = Transition(1.5, 0.5)
print(*dataclasses.astuple(t))
