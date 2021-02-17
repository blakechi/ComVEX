import os
import sys
sys.path.insert(0, os.getcwd())

import torch
from torchvision import transforms, datasets

from models.ViT.vision_transformer import ViT

EPOCH = 10
BATCH_SIZE = 32
LR_RATE = 1e-4

if __name__ == "__main__":

    train_dataset = datasets.MNIST(
        'datasets', 
        train=True, 
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    )

    test_dataset = datasets.MNIST(
        'datasets', 
        train=False, 
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    for epoch in range(EPOCH):
        for batch_id, (data, y) in enumerate(train_loader):
            ...