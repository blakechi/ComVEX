import torch
from torch import nn
from einops import rearrange

from Transformer.transformer import Transformer
from utils.helpers import Residual, Norm, FeedForward


class ViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        num_classes,
        dim,
        num_heads,
        num_layers,
        feadforward_dim,
        dropout=0.0,
        proj_head_dropout=0.0,
        *,
        self_defined_transformer=None,
    ):
        super(ViT, self).__init__()

        self.num_patches = (img_size / patch_size) ** 2
        patch_dim = patch_size * patch_size * 3

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert (
            self.head_dim * self.num_heads == self.dim
        ), "Token dimensions must be divided by the number of heads"

        self.linear_proj_patches = nn.Linear(patch_dim, dim, bias=False)
        self.CLS = nn.Parameter(torch.randn(1, 1, dim))
        self.position_code = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.token_dropout = nn.Dropout(dropout)

        self.transformer = (
            self_defined_transformer
            if self_defined_transformer is not None
            else Transformer(dim, num_heads, num_layers, feadforward_dim, dropout)
        )

        self.proj_head = nn.Sequential(
            Residual(
                Norm(
                    nn.Sequential(
                        nn.Linear(dim, dim),
                        nn.GELU(),
                    ),
                    dim=dim
                )
            ),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, att_mask=None, padding_mask=None):
        b, c, h, w, p = *x.shape, self.num_patches

        # Divide into flattened patches
        x = rearrange(x, "b c (h p) (w p_) -> b (h w) (p p_ c)", p=p, p_=p)

        # Linear projection
        x = self.linear_proj_patches(x)
        # Prepend CLS token and add position code
        CLS = repeat(self.CLS, "1 1 d -> b 1 d", b=b)
        x = torch.cat([CLS, x], dim=1) + self.position_code
        # Token dropout
        x = self.token_dropout(x)
        
        # Transformer
        x = self.transformer(x)

        # Projection head
        cls_output = x[:, 0, :]
        x = self.proj_head(x)

        return x



        