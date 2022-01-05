import torch
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def run():
    training_transforms = transforms.Compose([ToTensor()])
    training_data = datasets.MNIST(root="data", train=True, download=True, transform=training_transforms)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
    training_loader = DataLoader(training_data, 64, shuffle=True)

    c = 0
    for inputs, labels in training_loader:
        c += 1

    print(c)


run()
