from typing import Optional

from torch import nn
from comvex.utils import ConfigBase


class ViTConfig(ConfigBase):
    def __init__(
        self,
        image_channel: int,
        image_size: int,  # one lateral's size of a squre image
        patch_size: int,  # one lateral's size of a squre patch
        dim: int,  # tokens' dimension
        depth: int,
        num_heads: int,
        num_classes: int,
        pred_act_fnc_name: str = "ReLU",
        pre_norm: bool = False,
        ff_dim: Optional[int] = None,  # If not specify, ff_dim = 4*dim
        ff_dropout: float = 0.0,
        token_dropout: float = 0.0,
        self_defined_transformer: Optional[nn.Module] = None,    
    ) -> None:
        super().__init__()

        self.image_channel = image_channel
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.pred_act_fnc_name = pred_act_fnc_name
        self.pre_norm = pre_norm
        self.ff_dim = ff_dim
        self.ff_dropout = ff_dropout
        self.token_dropout = token_dropout
        self.self_defined_transformer = self_defined_transformer
        
    @classmethod
    def ViT_B(cls, num_classes, **kwargs):
        return cls(
            3,
            224,
            16,
            768,
            12,
            12,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def ViT_L(cls, num_classes, **kwargs):
        return cls(
            3,
            224,
            16,
            1024,
            24,
            16,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def ViT_B(cls, num_classes, **kwargs):
        return cls(
            3,
            224,
            16,
            1280,
            32,
            16,
            ff_dim=3072,
            num_classes=num_classes,
            **kwargs
        )