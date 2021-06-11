from comvex.utils import ConfigBase


class FNetConfig(ConfigBase):
    def __init__(
        self,
        image_size: int, 
        image_channel: int, 
        patch_size: int,
        dim: int,
        depth: int,
        num_classes: int,
        pre_norm: bool = False,
        ff_dim: int = None,
        ff_dropout: float = 0.0,
        token_dropout: float = 0.0,
        ff_act_fnc_name: str = "ReLU",
        dense_act_fnc_name: str = "ReLU",
        pred_act_fnc_name: str = "ReLU",
    ) -> None:

        self.image_size = image_size
        self.image_channel = image_channel
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_classes = num_classes 
        self.pre_norm = pre_norm
        self.ff_dim = ff_dim
        self.ff_dropout = ff_dropout
        self.token_dropout = token_dropout
        self.ff_act_fnc_name = ff_act_fnc_name
        self.dense_act_fnc_name = dense_act_fnc_name
        self.pred_act_fnc_name = pred_act_fnc_name

    @classmethod
    def FNet_L_24(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            1024,
            24,
            num_classes=num_classes,
            ff_dim=4096,
            **kwargs
        )

    @classmethod
    def FNet_B_12_768(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            768,
            12,
            num_classes=num_classes,
            ff_dim=3072,
            **kwargs
        )

    @classmethod
    def FNet_B_12_512(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            512,
            12,
            num_classes=num_classes,
            ff_dim=2048,
            **kwargs
        )

    @classmethod
    def FNet_B_8_512(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            512,
            8,
            num_classes=num_classes,
            ff_dim=2048,
            **kwargs
        )
        
    @classmethod
    def FNet_Mini_4_512(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            512,
            4,
            num_classes=num_classes,
            ff_dim=2048,
            **kwargs
        )

    @classmethod
    def FNet_Mini_4_256(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            256,
            4,
            num_classes=num_classes,
            ff_dim=1024,
            **kwargs
        )

    @classmethod
    def FNet_Micro_2_256(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            256,
            2,
            num_classes=num_classes,
            ff_dim=1024,
            **kwargs
        )

    @classmethod
    def FNet_Micro_2_128(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            128,
            2,
            num_classes=num_classes,
            ff_dim=512,
            **kwargs
        )
