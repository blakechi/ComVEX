import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from comvex.transformer import Transformer
from comvex.utils import FeedForward


class ViTBase(nn.Module):
    def __init__(self, image_size, image_channel, patch_size):
        super().__init__()

        assert image_size is not None, f"[{self.__class__.__name__}] Please specify input images' size."
        assert patch_size is not None, f"[{self.__class__.__name__}] Please specify patches' size."

        self.patch_size = patch_size
        self.patch_dim = (patch_size**2) * image_channel
        self.num_patches = (image_size // patch_size) ** 2

        assert (
            (self.num_patches**0.5) * patch_size == image_size
        ), f"[{self.__class__.__name__}] Image size must be divided by the patch size."

        self.patch_and_flat = Rearrange("b c (h p) (w q) -> b (h w) (p q c)", p=self.patch_size, q=self.patch_size)


class ViT(ViTBase):
    def __init__(
        self,
        image_size,    # one lateral's size of a squre image
        image_channel,
        patch_size,  # one lateral's size of a squre patch
        num_classes,
        *,
        dim,         # tokens' dimension
        num_heads,
        depth,
        pre_norm=False,
        ff_dim=None,                    # If not specify, ff_dim = 4*dim
        ff_dropout=0.0,
        token_dropout=0.0,
        self_defined_transformer=None,
    ):
        super().__init__(image_size, image_channel, patch_size)

        self.linear_proj = nn.Linear(self.patch_dim, dim, bias=False)
        self.CLS = nn.Parameter(torch.randn(1, 1, dim))
        self.position_code = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))  # plus 1 for CLS
        self.token_dropout = nn.Dropout(token_dropout)

        self.transformer = (
            self_defined_transformer
            if self_defined_transformer is not None
            else Transformer(
                dim=dim, 
                depth=depth,
                heads=num_heads,
                pre_norm=pre_norm,
                ff_dim=ff_dim,
                ff_dropout=ff_dropout,
            )
        )

        self.proj_head = FeedForward(
            dim=dim,
            hidden_dim=dim,
            output_dim=num_classes,
        )

    def forward(self, x, attention_mask=None, padding_mask=None):
        b, c, h, w, p = *x.shape, self.num_patches

        # Divide into flattened patches
        x = self.patch_and_flat(x)

        # Linear projection
        x = self.linear_proj(x)

        # Token dropout
        x = self.token_dropout(x)

        # Prepend CLS token and add position code
        CLS = repeat(self.CLS, "1 1 d -> b 1 d", b=b)
        x = torch.cat([CLS, x], dim=1) + self.position_code
        
        # Transformer
        x = self.transformer(x, attention_mask, padding_mask)

        # Projection head
        cls_output = x[:, 0, :]
        
        return self.proj_head(cls_output)


        