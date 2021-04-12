import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from models.perceiver import Perceiver


if __name__ == "__main__":

    perceiver = Perceiver(
        data_shape=[1, 224, 224],
        cross_heads=1,
        num_latent_tokens=1024,
        dim=512, 
        heads=16, 
        layers_indice=[0] + [1]*7, 
        num_latent_transformers_in_layers=[6]*2, 
        num_bands=64,
        resolution=224,
        frequency_base=2,
        pre_norm=True,
        ff_dim=None, 
        ff_dim_scale=4, 
        ff_dropout=0.0,
        attention_dropout=0.0,
        cross_kv_dim=None,
        head_dim=None
    )

    print(perceiver)

    x = torch.randn(1, 1, 224, 224)

    print("Input Shape:\n", x.shape)
    print("Output Shape:\n", perceiver(x).shape)