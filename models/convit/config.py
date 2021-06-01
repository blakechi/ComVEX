class ConViTConfig(object):
    def __init__(
        self,
        image_size: int, 
        image_channel: int, 
        patch_size: int,
        num_local_layers: int,
        num_nonlocal_layers: int,
        dim: int,
        num_classes: int,
        locality_strength=None,
        heads=None,
        head_dim=None,
        pre_norm=False,
        ff_dim=None,                    # If not specify, ff_dim = 4*dim
        ff_dropout=0.0,
        attention_dropout: float = 0.0,
        token_dropout: float = 0.0,
        pred_act_fnc_name: str = "ReLU",
    ) -> None:

        self.image_size = image_size
        self.image_channel = image_channel
        self.patch_size = patch_size
        self.num_local_layers = num_local_layers
        self.num_nonlocal_layers = num_nonlocal_layers
        self.dim = dim
        self.num_classes = num_classes 
        self.locality_strength = locality_strength 
        self.heads = heads 
        self.head_dim = head_dim 
        self.pre_norm = pre_norm 
        self.ff_dim = ff_dim 
        self.ff_dropout = ff_dropout 
        self.attention_dropout = attention_dropout
        self.token_dropout = token_dropout
        self.pred_act_fnc_name = pred_act_fnc_name

    @classmethod
    def ConViT_Ti(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            10,
            2,
            192,
            num_classes=num_classes,
            locality_strength=1.0,
            heads=4,
            **kwargs
        )

    @classmethod
    def ConViT_Ti_plus(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            10,
            2,
            192,
            num_classes=num_classes,
            locality_strength=1.0,
            heads=4,
            **kwargs
        )
        
    @classmethod
    def ConViT_S(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            10,
            2,
            432,
            num_classes=num_classes,
            locality_strength=1.0,
            heads=9,
            **kwargs
        )
        
    @classmethod
    def ConViT_S_plus(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            10,
            2,
            576,
            num_classes=num_classes,
            locality_strength=1.0,
            heads=9,
            **kwargs
        )
        
    @classmethod
    def ConViT_B(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            10,
            2,
            768,
            num_classes=num_classes,
            locality_strength=1.0,
            heads=16,
            **kwargs
        )
        
    @classmethod
    def ConViT_B_plus(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            10,
            2,
            1024,
            num_classes=num_classes,
            locality_strength=1.0,
            heads=16,
            **kwargs
        )
        