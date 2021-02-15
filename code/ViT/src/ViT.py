import torch
from torch import nn
from einops import rearrange


class ViT(nn.modules):
    def __init__(
        self,
        # Image
        img_size,
        patch_size,
        num_classes,
        # Transformer
        dim,
        num_heads,
        depth,
        feadforward_dim,
        dropout,
        # Projection Head
        proj_head_dropout=0.1,
        *,
        # Others
        transformer=None,
    ):
        super(ViT, self).__init__()

        self.num_patches = (img_size/patch_size)**2
        patch_dim = patch_size*patch_size*3

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert (
            self.head_dim * self.num_heads == self.dim
        ), "Token dimensions must be divided by the number of heads"

        self.CLS = nn.Parameter(torch.randn(1, 1, dim))
        self.position_encoder = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.linear_proj_patches = nn.Linear(patch_dim, dim, bias=False)
        self.transformer = (
            transformer
            if transformer
            else Transformer(dim, num_heads, depth, feadforward_dim, dropout)
        )

        self.proj_head = nn.Linear(dim, num_classes, bias=)