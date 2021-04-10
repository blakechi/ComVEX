import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

from models.transformer import Transformer
from models.utils import FeedForward


class ViTBase(nn.Module):
    def __init__(self, image_size, image_channel, patch_size, dim, num_heads, head_dim=None):
        super().__init__()

        assert image_size is not None, f"[{self.__class__.__name__}] Please specify input images' size."
        assert patch_size is not None, f"[{self.__class__.__name__}] Please specify patches' size."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = (patch_size**2) * image_channel
        self.patch_size = patch_size

        assert (
            self.head_dim * self.num_heads == self.dim
        ), "Token dimensions must be divided by the number of heads"

        assert (
            (self.num_patches**0.5) * patch_size == image_size
        ), f"[{self.__class__.__name__}] Image size must be divided by the patch size."

        self.flatten_to_patch = Rearrange("b c (h p) (w q) -> b (h w) (p q c)", p=self.patch_size, q=self.patch_size)


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
        num_layers,
        ff_dim=None,
        ff_dropout=0.0,
        token_dropout=0.0,
        self_defined_transformer=None,
    ):
        super().__init__(image_size, image_channel, patch_size, dim, num_heads)

        self.linear_proj_patches = nn.Linear(self.patch_dim, self.dim, bias=False)
        self.CLS = nn.Parameter(torch.randn(1, 1, self.dim))
        self.position_code = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))  # plus 1 for CLS
        self.token_dropout = nn.Dropout(token_dropout)

        self.transformer = (
            self_defined_transformer
            if self_defined_transformer is not None
            else Transformer(
                dim=self.dim, 
                heads=self.num_heads,
                head_dim=self.head_dim,
                depth=num_layers,
                ff_dim=ff_dim if ff_dim is not None else self.dim*4,
                ff_dropout=ff_dropout,
                max_seq_len=self.num_patches,
            )
        )

        self.proj_head = FeedForward(
            dim=self.dim,
            hidden_dim=self.dim,
            output_dim=num_classes,
            useResidualWithNorm=True,
        )

    def forward(self, x, att_mask=None, padding_mask=None):
        b, c, h, w, p = *x.shape, self.num_patches

        # Divide into flattened patches
        x = self.flatten_to_patch(x)

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
        cls_output = x[..., 0, :]
        
        return self.proj_head(cls_output)


        