import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time

device = "cuda"


class Trainer:
    def __init__(self, batch_size, num_features):
        self.batch_size = batch_size
        img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(mean=[0.456],
                                 std=[0.224]),
            transforms.Lambda(lambda t: t.repeat(3, 1, 1))

        ])
        training_data = datasets.MNIST(root="data", train=True, download=True, transform=img_transforms)
        validation_data = datasets.MNIST(root="data", train=False, download=True, transform=img_transforms)

        self.data_loaders = {
            "train": DataLoader(training_data, self.batch_size, shuffle=True),
            "val": DataLoader(validation_data, self.batch_size, shuffle=True)}

        self.model = models.resnet18(pretrained=True)
        # Freeze everything except last fc layer
        for param in self.model.parameters():
            param.requires_grad = False

        # self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=num_features)

        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=3e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self):
        for phase in ["train"]:
            for epoch in range(5):
                running_correct = 0
                data_loader = self.data_loaders[phase]
                num_batches = 0
                end = time.time()
                for batch_nr, (images, labels) in enumerate(data_loader):
                    num_batches += 1
                    images = images.to(device)
                    labels = labels.to(device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    start = end
                    end = time.time()

                    _, predictions = torch.max(outputs, 1)
                    correct = torch.sum(predictions == labels) / self.batch_size
                    running_correct += correct
                    if batch_nr % 20 == 0:
                        print(correct, "ms: " + str((end - start) * 1000))

                print(f"Epoch accuracy: {running_correct / num_batches}")

    def test_model(self):
        img, label = next(iter(self.data_loaders["val"]))
        with torch.no_grad():
            print(img[0])


if __name__ == "__main__":
    trainer = Trainer(batch_size=64, num_features=10)
    trainer.train()
    # trainer.test_model()
