import torch
from torch import nn
from einops import rearrange

from models.Transformer.transformer import Transformer
from models.utils.base_block import Residual, Norm, FeedForward


class ViT(nn.Module):
    def __init__(
        self,
        img_size,    # one lateral's size of a squre image
        patch_size,  # one lateral's size of a squre patch
        num_classes,
        dim,         # tokens' dimension
        num_heads,
        num_layers,
        feedforward_dim,
        dropout=0.0,
        proj_head_dropout=0.0,
        *,
        self_defined_transformer=None,
    ):
        super(ViT, self).__init__()

        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * 3

        assert (
            (self.num_patches**0.5) * patch_size == img_size
        ), "Token dimensions must be divided by the number of heads"

        dim = dim
        num_heads = num_heads
        head_dim = dim // num_heads

        assert (
            head_dim * num_heads == dim
        ), "Token dimensions must be divided by the number of heads"

        self.linear_proj_patches = nn.Linear(patch_dim, dim, bias=False)
        self.CLS = nn.Parameter(torch.randn(1, 1, dim))
        self.position_code = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.token_dropout = nn.Dropout(dropout)

        self.transformer = (
            self_defined_transformer
            if self_defined_transformer is not None
            else Transformer(
                dim=dim, 
                heads=num_heads,
                depth=num_layers,
                ff_dim=feedforward_dim ,
                ff_dropout=dropout,
                max_seq_len=self.num_patches
            )
        )

        self.proj_head = FeedForward(
            dim=dim,
            hidden_dim=dim,
            output_dim=num_classes,
            dropout=proj_head_dropout,
            useResidualWithNorm=True,
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
        
        return self.proj_head(cls_output)


if __name__ == "__main__":
    
    vit = ViT(
        img_size=1024,
        patch_size=32,
        num_classes=6,
        dim=64,
        num_heads=8,
        num_layers=12,
        feedforward_dim=128,
    )


        