import torch
from torch import nn
from einops import rearrange

from models.transformer import Transformer
from models.utils import FeedForward


class ViTBase(nn.Module):
    def __init__(self, image_size, patch_size, dim, num_heads, image_channel=3, head_dim=None):
        super().__init__()

        assert image_size is not None, f"[{self.__class__.__name__}] Please specify input images' size."
        assert patch_size is not None, f"[{self.__class__.__name__}] Please specify patches' size."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * image_channel
        self.patch_size = patch_size

        assert (
            self.head_dim * self.num_heads == self.dim
        ), "Token dimensions must be divided by the number of heads"

        assert (
            (self.num_patches**0.5) * patch_size == image_size
        ), f"[{self.__class__.__name__}] Image size must be divided by the patch size."


class ViT(ViTBase):
    def __init__(
        self,
        image_size,    # one lateral's size of a squre image
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
        super().__init__(image_size, patch_size, dim, num_heads)

        self.linear_proj_patches = nn.Linear(self.patch_dim, self.dim, bias=False)
        self.CLS = nn.Parameter(torch.randn(1, 1, self.dim))
        self.position_code = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
        self.token_dropout = nn.Dropout(dropout)

        self.transformer = (
            self_defined_transformer
            if self_defined_transformer is not None
            else Transformer(
                dim=self.dim, 
                head_dim=self.head_dim,
                depth=num_layers,
                ff_dim=feedforward_dim ,
                ff_dropout=dropout,
                max_seq_len=self.num_patches,
            )
        )

        self.proj_head = FeedForward(
            dim=self.dim,
            hidden_dim=self.dim,
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
        image_size=1024,
        patch_size=32,
        num_classes=6,
        dim=64,
        num_heads=8,
        num_layers=12,
        feedforward_dim=128,
    )


        