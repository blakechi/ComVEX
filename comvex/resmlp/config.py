class ResMLPConfig(object):
    def __init__(
        self,
        image_size: int, 
        image_channel: int, 
        patch_size: int, 
        depth: int, 
        dim: int, 
        num_classes: int,
        path_dropout: float = 0., 
        token_dropout: float = 0.,
        ff_dropout: float = 0.
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.image_channel = image_channel
        self.patch_size = patch_size
        self.depth = depth
        self.dim = dim
        self.num_classes = num_classes
        self.path_dropout = path_dropout
        self.token_dropout = token_dropout
        self.ff_dropout = ff_dropout

    @classmethod
    def ResMLP_12(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            12,
            384,
            num_classes,
            **kwargs
        )
    
    @classmethod
    def ResMLP_24(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            24,
            384,
            num_classes,
            **kwargs
        )
    
    @classmethod
    def ResMLP_36(cls, num_classes, **kwargs):
        return cls(
            224,
            3,
            16,
            36,
            384,
            num_classes,
            **kwargs
        )