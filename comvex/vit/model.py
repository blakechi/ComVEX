from typing import Optional
from einops.einops import rearrange
import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange

from comvex.transformer import Transformer
from comvex.utils import ProjectionHead, TokenDropout, ClassMultiheadAttention
from comvex.utils.helpers import name_with_msg, config_pop_argument
from .config import ViTConfig


class ViTBase(nn.Module):
    def __init__(self, image_size, image_channel, patch_size, use_patch_and_flat=True):
        super().__init__()

        assert image_size is not None, name_with_msg(self, "Please specify input images' size")
        assert patch_size is not None, name_with_msg(self, "Please specify patches' size")

        self.patch_size = patch_size
        self.patch_dim = (patch_size**2) * image_channel
        self.num_patches = (image_size // patch_size) ** 2

        assert (
            (self.num_patches**0.5) * patch_size == image_size
        ), name_with_msg(self, "Image size must be divided by the patch size")
        
        if use_patch_and_flat:
            self.patch_and_flat = Rearrange("b c (h p) (w q) -> b (h w) (p q c)", p=self.patch_size, q=self.patch_size)


class ViTBackbone(ViTBase):
    def __init__(
        self,
        image_channel,
        image_size,  # one lateral's size of a squre image
        patch_size,  # one lateral's size of a squre patch
        dim,  # tokens' dimension
        num_heads,
        depth,
        use_multihead_attention_pooling: bool = True,
        cat_cls_to_context: bool = False,
        pre_norm=False,
        ff_dim=None,  # If not specify, ff_dim = 4*dim
        ff_dropout=0.0,
        attention_dropout=0.0,
        token_dropout=0.0,
        self_defined_transformer=None,
    ):
        super().__init__(image_size, image_channel, patch_size)

        self.linear_proj = nn.Linear(self.patch_dim, dim, bias=False)
        self.CLS = nn.Parameter(torch.randn(1, 1, dim))
        self.position_code = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))  # plus 1 for CLS
        self.token_dropout = TokenDropout(token_dropout)

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
                attention_dropout=attention_dropout
            )
        )
        
        self.use_multihead_attention_pooling = use_multihead_attention_pooling
        if self.use_multihead_attention_pooling:
            self.map = ClassMultiheadAttention(
                dim=dim,
                heads=num_heads,
                cat_cls_to_context_at_dim=1 if cat_cls_to_context else None,
                ff_dropout=ff_dropout,
                attention_dropout=attention_dropout
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
            )

    def forward(self, x):
        b, c, h, w = x.shape
        cls_tokens = repeat(self.CLS, "1 1 d -> b 1 d", b=b)

        # Divide into flattened patches
        x = self.patch_and_flat(x)

        # Linear projection
        x = self.linear_proj(x)

        # Token dropout
        x = self.token_dropout(x)

        # Concatenate CLS if not using multihead attention pooling
        if not self.use_multihead_attention_pooling:
            x = torch.cat([cls_tokens, x], dim=1) + self.position_code
        
        # Transformer
        x = self.transformer(x)

        # Use multihead attention pooling if specified
        if self.use_multihead_attention_pooling:
            cls_tokens = self.map(cls_tokens, x)
            cls_tokens = rearrange(cls_tokens, "b 1 d -> b d")
        else:
            cls_tokens = x.select(dim=1, index=0)
            cls_tokens = self.proj(cls_tokens)

        return cls_tokens


class ViTWithLinearClassifier(ViTBackbone):
    def __init__(self, config: ViTConfig) -> None:
        num_classes = config_pop_argument(config, "num_classes")
        pred_act_fnc_name = config_pop_argument(config, "pred_act_fnc_name")
        super().__init__(**config.__dict__)

        self.proj_head = ProjectionHead(
            dim=config.dim,
            out_dim=num_classes,
            act_fnc_name=pred_act_fnc_name
        )

    def forward(self, x):
        x = super().forward(x)
        
        return self.proj_head(x)
        