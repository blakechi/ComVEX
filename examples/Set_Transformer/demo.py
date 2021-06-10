import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from comvex.set_transformer import SetTransformer, ISAB


if __name__ == "__main__":
        
    set_transformer = SetTransformer(
        dim=128, 
        heads=4, 
        encoder_base_block=ISAB,
        num_inducing_points=16, 
        num_seeds=4, 
        attention_dropout=0.0, 
        ff_dropout=0.0, 
        ff_dim_scale=4, 
        pre_norm=False, 
        head_dim=None
    )

    print(set_transformer)

    x = torch.randn(1, 4, 128)

    print("Input Shape:\n", x.shape)
    print("Output Shape (remain the output size as the input):\n", set_transformer(x).shape)