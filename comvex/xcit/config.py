from typing import Optional

from comvex.utils import ConfigBase


class XCiTConfig(ConfigBase):
    def __init__(
        self,
        image_size: int,
        image_channel: int,
        patch_size: int,
        self_attn_depth: int,
        cls_attn_depth: int,
        dim: int,
        heads: int,
        num_classes: int,
        alpha: float = 1e-5,
        local_kernel_size: int = 3,
        act_fnc_name: str = "GELU",
        use_bias: bool = False,
        ff_expand_scale: int = 4,
        ff_dropout: float = 0.,
        attention_dropout: float = 0.,
        path_dropout: float = 0.,
        token_dropout: float = 0.,
        upsampling_mode: Optional[str] = None,
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
        self.num_classes = num_classes
        self.alpha = alpha
        self.local_kernel_size = local_kernel_size
        self.act_fnc_name = act_fnc_name
        self.use_bias = use_bias
        self.ff_expand_scale = ff_expand_scale
        self.ff_dropout = ff_dropout
        self.attention_dropout = attention_dropout
        self.path_dropout = path_dropout
        self.token_dropout = token_dropout
        self.upsampling_mode = upsampling_mode
        self.pred_act_fnc_name = pred_act_fnc_name

    @classmethod
    def XCiT_N12_224_16(cls, num_classes: int, **kwargs) -> "XCiTConfig":
        return cls(
            224,
            3,
            16,
            12,
            2,
            128,
            4,
            num_classes=num_classes
        )

    @classmethod
    def XCiT_N12_384_8(cls, num_classes: int, **kwargs) -> "XCiTConfig":
        return cls(
            384,
            3,
            8,
            12,
            2,
            128,
            4,
            num_classes=num_classes
        )

    @classmethod
    def XCiT_T12_224_16(cls, num_classes: int, **kwargs) -> "XCiTConfig":
        return cls(
            224,
            3,
            16,
            12,
            2,
            192,
            4,
            num_classes=num_classes
        )

    @classmethod
    def XCiT_T12_384_8(cls, num_classes: int, **kwargs) -> "XCiTConfig":
        return cls(
            384,
            3,
            8,
            12,
            2,
            192,
            4,
            num_classes=num_classes
        )

    @classmethod
    def XCiT_T24_224_16(cls, num_classes: int, **kwargs) -> "XCiTConfig":
        return cls(
            224,
            3,
            16,
            24,
            2,
            192,
            4,
            num_classes=num_classes
        )

    @classmethod
    def XCiT_T24_384_8(cls, num_classes: int, **kwargs) -> "XCiTConfig":
        return cls(
            384,
            3,
            8,
            24,
            2,
            192,
            4,
            num_classes=num_classes
        )

    @classmethod
    def XCiT_S12_224_16(cls, num_classes: int, **kwargs) -> "XCiTConfig":
        return cls(
            224,
            3,
            16,
            12,
            2,
            384,
            8,
            num_classes=num_classes
        )

    @classmethod
    def XCiT_S12_384_8(cls, num_classes: int, **kwargs) -> "XCiTConfig":
        return cls(
            384,
            3,
            8,
            12,
            2,
            384,
            8,
            num_classes=num_classes
        )

    @classmethod
    def XCiT_S24_224_16(cls, num_classes: int, **kwargs) -> "XCiTConfig":
        return cls(
            224,
            3,
            16,
            24,
            2,
            384,
            8,
            num_classes=num_classes
        )

    @classmethod
    def XCiT_S24_384_8(cls, num_classes: int, **kwargs) -> "XCiTConfig":
        return cls(
            384,
            3,
            8,
            24,
            2,
            384,
            8,
            num_classes=num_classes
        )

    @classmethod
    def XCiT_M24_224_16(cls, num_classes: int, **kwargs) -> "XCiTConfig":
        return cls(
            224,
            3,
            16,
            24,
            2,
            512,
            8,
            num_classes=num_classes
        )

    @classmethod
    def XCiT_M24_384_8(cls, num_classes: int, **kwargs) -> "XCiTConfig":
        return cls(
            384,
            3,
            8,
            24,
            2,
            512,
            8,
            num_classes=num_classes
        )

    @classmethod
    def XCiT_L24_224_16(cls, num_classes: int, **kwargs) -> "XCiTConfig":
        return cls(
            224,
            3,
            16,
            24,
            2,
            768,
            16,
            num_classes=num_classes
        )
    
    @classmethod
    def XCiT_L24_384_8(cls, num_classes: int, **kwargs) -> "XCiTConfig":
        return cls(
            384,
            3,
            8,
            24,
            2,
            768,
            16,
            num_classes=num_classes
        )
    