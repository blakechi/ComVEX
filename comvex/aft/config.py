from typing import Optional, Literal

from comvex.utils import ConfigBase


class AFTConfig(ConfigBase):
    def __init__(
        self,
        image_size: int,
        image_channel: int,
        patch_size: int,
        num_layers: int,
        dim: int,
        num_classes: int,
        local_window_size: Optional[int] = 0,
        hidden_dim: Optional[int] = None,
        aft_mode: Literal["full", "simple", "local", "conv", "general"] = "full",
        pool_mode: Literal["mean", "class"] = "mean",
        query_act_fnc_name: str = "Sigmoid",
        use_bias: bool = False,
        ff_expand_scale: int = 4,
        ff_dropout: float = 0.,
        attention_dropout: float = 0.,
        path_dropout: float = 0.,
        # AFT - General, Full, Simple, Local
        position_bias_dim: int = 128,
        use_position_bias: bool = True,
        # AFT - Conv
        heads: int = 32,
        epsilon: float = 1e-8,
        # Possible Class Attention Layer
        alpha: float = 1e-5,
        cls_attn_heads: int = 16,
        # Projection Head
        pred_act_fnc_name: str = "ReLU"
    ) -> None:
        super().__init__()

        self.image_size=image_size
        self.image_channel=image_channel
        self.patch_size=patch_size
        self.num_layers=num_layers
        self.dim=dim
        self.local_window_size=local_window_size
        self.num_classes=num_classes
        self.hidden_dim=hidden_dim
        self.aft_mode=aft_mode
        self.pool_mode=pool_mode
        self.query_act_fnc_name=query_act_fnc_name
        self.use_bias=use_bias
        self.ff_expand_scale=ff_expand_scale
        self.ff_dropout=ff_dropout
        self.attention_dropout=attention_dropout
        self.path_dropout=path_dropout
        # AFT - General, Full, Simple, Local
        self.position_bias_dim=position_bias_dim
        self.use_position_bias=use_position_bias
        # AFT - Conv
        self.heads=heads
        self.epsilon=epsilon
        # Possible Class Attention Layer
        self.alpha=alpha
        self.cls_attn_heads=cls_attn_heads
        self.pred_act_fnc_name=pred_act_fnc_name

    @classmethod
    def AFT_Full_tiny(cls, num_classes: int, **kwargs) -> "AFTConfig":
        return cls(
            224,
            3,
            16,
            num_layers=12,
            dim=192,
            aft_mode="full",
            pool_mode="mean",
            position_bias_dim=128,
            local_window_size=None,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def AFT_Full_small(cls, num_classes: int, **kwargs) -> "AFTConfig":
        return cls(
            224,
            3,
            16,
            num_layers=12,
            dim=384,
            aft_mode="full",
            pool_mode="mean",
            position_bias_dim=128,
            local_window_size=None,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def AFT_Conv_tiny_32_11(cls, num_classes: int, **kwargs) -> "AFTConfig":
        return cls(
            224,
            3,
            16,
            num_layers=12,
            dim=192,
            heads=32,
            local_window_size=11,
            aft_mode="conv",
            pool_mode="mean",
            position_bias_dim=128,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def AFT_Conv_tiny_192_11(cls, num_classes: int, **kwargs) -> "AFTConfig":
        return cls(
            224,
            3,
            16,
            num_layers=12,
            dim=192,
            heads=192,
            local_window_size=11,
            aft_mode="conv",
            pool_mode="mean",
            position_bias_dim=128,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def AFT_Conv_small_16_11(cls, num_classes: int, **kwargs) -> "AFTConfig":
        return cls(
            224,
            3,
            16,
            num_layers=12,
            dim=384,
            heads=16,
            local_window_size=11,
            aft_mode="conv",
            pool_mode="mean",
            position_bias_dim=128,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def AFT_Conv_small_384_11(cls, num_classes: int, **kwargs) -> "AFTConfig":
        return cls(
            224,
            3,
            16,
            num_layers=12,
            dim=384,
            heads=384,
            local_window_size=11,
            aft_mode="conv",
            pool_mode="mean",
            position_bias_dim=128,
            num_classes=num_classes,
            **kwargs
        )

    @classmethod
    def AFT_Conv_small_384_15(cls, num_classes: int, **kwargs) -> "AFTConfig":
        return cls(
            224,
            3,
            16,
            num_layers=12,
            dim=384,
            heads=384,
            local_window_size=15,
            aft_mode="conv",
            pool_mode="mean",
            position_bias_dim=128,
            num_classes=num_classes,
            **kwargs
        )