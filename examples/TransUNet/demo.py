import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from comvex.transunet import TransUNet


if __name__ == "__main__":
    
    transUnet = TransUNet(
        input_channel=3,
        middle_channel=512,
        output_channel=2,
        channel_in_between=[64, 128, 256],
        num_res_blocks_in_between=[3, 4, 9],
        image_size=224,
        patch_size=2,
        dim=512,
        num_heads=16,
        num_layers=12,
        token_dropout=0,
        ff_dropout=0,
        to_remain_size=True
    )
    print(transUnet)

    x = torch.randn(1, 3, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", transUnet(x).shape)