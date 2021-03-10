import os
import sys
sys.path.insert(0, os.getcwd())

import torch
from torchvision import transforms, datasets

from models.vit import ViT
from models.transunet import TransUNet

EPOCH = 10
BATCH_SIZE = 32
LR_RATE = 1e-4


if __name__ == "__main__":

    transUnet = TransUNet(
        input_channel=3,
        middle_channel=1024,
        output_channel=10,
        patch_size=16,
        image_size=512,
        vit_dim=512,
        vit_num_heads=16,
        vit_num_layers=12,
        vit_feedforward_dim=2048,
        vit_dropout=0,
        channel_in_between=[64, 128, 256],
        to_remain_size=True
    )
    print(transUnet)

    x = torch.randn(1, 3, 512, 512)

    print(transUnet(x).shape)

    assert False

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
            print(data.shape)
            print(y.shape)