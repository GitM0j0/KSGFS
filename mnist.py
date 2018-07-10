import torch
import torchvision
import os

from torchvision import transforms

def get_mnist():
    train_data = torchvision.datasets.MNIST(root=os.environ.get("DATASETS_PATH", "~/datasets"), train=True,
                                             download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=5)

    test_data = torchvision.datasets.MNIST(root=os.environ.get("DATASETS_PATH", "~/datasets"), train=False,
                                            download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

    return train_loader, test_loader
