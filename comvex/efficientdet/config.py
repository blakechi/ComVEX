from typing import Optional, List, Literal

from comvex.utils import EfficientNetConfig


class EfficientDetConfig(EfficientNetConfig):
    def __init__(
        self,
        image_channel: int,
        depth_scale: float,
        width_scale: float,
        resolution: int,
        num_classes: int,
        up_sampling_mode: Optional[str] = None,
        act_fnc_name: str = "SiLU",
        se_act_fnc_name: str = "SiLU",
        se_scale: float = 0.25,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.99,
        return_feature_maps: bool = False,
        path_dropout: float = 0.2,
        ff_dropout: float = 0.,
    ) -> None:
        super().__init__()

        self.image_channel = image_channel
        self.depth_scale = depth_scale
        self.width_scale = width_scale
        self.resolution = resolution
        self.num_classes = num_classes
        self.up_sampling_mode = up_sampling_mode
        self.act_fnc_name = act_fnc_name
        self.se_act_fnc_name = se_act_fnc_name
        self.se_scale = se_scale
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.return_feature_maps = return_feature_maps
        self.path_dropout = path_dropout
        self.ff_dropout = ff_dropout
    