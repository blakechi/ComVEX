class gMLPConfig(object):
    def __init__(
        self,
        image_size: int, 
        image_channel: int, 
        patch_size: int,
        depth: int,
        dim: int,
        ffn_dim: int,
        num_classes: int,
        pred_act_fnc_name: str = "ReLU",
        attention_dim: int = None,
        attention_dropout: float = 0.0,
        token_dropout: float = 0.0,
    ) -> None:

        self.image_size = image_size
        self.image_channel = image_channel
        self.patch_size = patch_size
        self.depth = depth
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_classes = num_classes 
        self.pred_act_fnc_name = pred_act_fnc_name
        self.token_dropout = token_dropout
        self.attention_dropout = attention_dropout
        self.attention_dim = attention_dim

    @classmethod
    def gMLP_Ti(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            30,
            128,
            768,
            num_classes=num_classes,
            **kwargs
        )
        
    @classmethod
    def gMLP_S(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            30,
            256,
            1536,
            num_classes=num_classes,
            **kwargs
        )
        
    @classmethod
    def gMLP_B(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            30,
            512,
            3072,
            num_classes=num_classes,
            **kwargs
        )
