from typing import Optional

from comvex.utils import ConfigBase


class CaiTConfig(ConfigBase):
    def __init__(
        self,
        image_size: int,
        image_channel: int,
        patch_size: int,
        self_attn_depth: int,
        cls_attn_depth: int,
        dim: int,
        alpha: float,
        num_classes: int,
        heads: Optional[int] = None,
        ff_expand_scale: int = 4,
        ff_dropout: float = 0.,
        token_dropout: float = 0.,
        attention_dropout: float = 0.,
        path_dropout: float = 0.,
        pred_act_fnc_name: str = "ReLU",
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.image_channel = image_channel
        self.patch_size = patch_size
        self.self_attn_depth = self_attn_depth
        self.cls_attn_depth = cls_attn_depth
        self.dim = dim
        self.heads = heads
        self.alpha = alpha
        self.num_classes = num_classes
        self.ff_expand_scale = ff_expand_scale
        self.ff_dropout = ff_dropout
        self.token_dropout = token_dropout
        self.attention_dropout = attention_dropout
        self.path_dropout = path_dropout
        self.pred_act_fnc_name = pred_act_fnc_name

    @classmethod
    def CaiT_XXS_24(cls, image_size: int, num_classes: int, **kwargs) -> "CaiTConfig":
        return cls(
            image_size,
            3,
            16,
            24,
            2,
            192,
            4,
            1e-5,
            path_dropout=0.05,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def CaiT_XXS_36(cls, image_size: int, num_classes: int, **kwargs) -> "CaiTConfig":
        return cls(
            image_size,
            3,
            16,
            36,
            2,
            192,
            4,
            1e-6,
            path_dropout=0.1,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def CaiT_XS_24(cls, image_size: int, num_classes: int, **kwargs) -> "CaiTConfig":
        return cls(
            image_size,
            3,
            16,
            24,
            2,
            288,
            6,
            1e-5,
            path_dropout=0.05,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def CaiT_XS_36(cls, image_size: int, num_classes: int, **kwargs) -> "CaiTConfig":
        return cls(
            image_size,
            3,
            16,
            36,
            2,
            288,
            6,
            1e-6,
            path_dropout=0.1,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def CaiT_S_24(cls, image_size: int, num_classes: int, **kwargs) -> "CaiTConfig":
        return cls(
            image_size,
            3,
            16,
            24,
            2,
            384,
            8,
            1e-5,
            path_dropout=0.1,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def CaiT_S_36(cls, image_size: int, num_classes: int, **kwargs) -> "CaiTConfig":
        return cls(
            image_size,
            3,
            16,
            36,
            2,
            384,
            8,
            1e-6,
            path_dropout=0.2,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def CaiT_S_48(cls, image_size: int, num_classes: int, **kwargs) -> "CaiTConfig":
        return cls(
            image_size,
            3,
            16,
            48,
            2,
            384,
            8,
            1e-6,
            path_dropout=0.3,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def CaiT_M_24(cls, image_size: int, num_classes: int, **kwargs) -> "CaiTConfig":
        return cls(
            image_size,
            3,
            16,
            24,
            2,
            768,
            16,
            1e-5,
            path_dropout=0.2,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def CaiT_M_36(cls, image_size: int, num_classes: int, **kwargs) -> "CaiTConfig":
        return cls(
            image_size,
            3,
            16,
            36,
            2,
            768,
            16,
            1e-6,
            path_dropout=0.3,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def CaiT_M_48(cls, image_size: int, num_classes: int, **kwargs) -> "CaiTConfig":
        return cls(
            image_size,
            3,
            16,
            48,
            2,
            768,
            16,
            1e-6,
            path_dropout=0.4,
            num_classes=num_classes,
            **kwargs
        )