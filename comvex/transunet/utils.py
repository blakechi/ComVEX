import torch
from torch import nn

from comvex.utils import ResNetFullPreActivationBottleneckBlock
from comvex.vit import ViTBase
from comvex.transformer import Transformer


class TransUNetEncoderConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_res_blocks):
        super().__init__()

        self.layers = nn.ModuleList([
            ResNetFullPreActivationBottleneckBlock(
                in_channel, 
                out_channel, 
                out_channel, 
                stride=2, 
                padding=1
            ) if idx == 0 else ResNetFullPreActivationBottleneckBlock(
                out_channel, 
                out_channel, 
                out_channel
            ) for idx in range(num_res_blocks)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class TransUNetViT(ViTBase):
    def __init__(
        self,
        image_size,
        image_channel,
        *,
        patch_size,   # one lateral's size of a squre patch
        dim,
        num_heads,
        num_layers,
        token_dropout=0,
        ff_dropout=0,
        ff_dim=None,
        self_defined_transformer=None,
        ):
        super().__init__(image_size, image_channel, patch_size)

        self.proj_patches = nn.Linear(self.patch_dim, dim, bias=False)

        # According to line 149: https://github.com/Beckschen/TransUNet/blob/main/networks/vit_seg_modeling.py#L149
        self.position_code = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.token_dropout = nn.Dropout(token_dropout)

        ff_dim = ff_dim if ff_dim is not None else 4*dim

        self.transformer = (
            self_defined_transformer
            if self_defined_transformer is not None
            else Transformer(
                dim=dim, 
                heads=num_heads,
                depth=num_layers,
                ff_dim=ff_dim,
                ff_dropout=ff_dropout,
                max_seq_len=self.num_patches
            )
        )

    def forward(self, x, attention_mask=None, padding_mask=None):
        b, c, h, w, p = *x.shape, self.num_patches

        # Images patching and projection
        x = self.patch_and_flat(x)
        x = self.proj_patches(x)

        # Add position code
        x = x + self.position_code
        
        # Token dropout
        x = self.token_dropout(x)
        
        # Transformer
        x = self.transformer(x, attention_mask, padding_mask)
        
        return x