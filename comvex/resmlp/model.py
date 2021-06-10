from typing import OrderedDict
import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce

from .config import ResMLPConfig
from comvex.vit import ViTBase
from comvex.utils import MLP, AffineTransform, PathDropout, LayerScaleBlock, TokenWiseDropout


class ResMLPLayer(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_patches: int, 
        alpha: float, 
        path_dropout=0., 
        ff_dropout=0.
    ):
        super().__init__()

        self.token_mixer = nn.Sequential(OrderedDict([
            ("aff_pre_norm", AffineTransform(dim)),
            ("transpose_0", Rearrange("b n d -> b d n")),
            ("linear", nn.Linear(num_patches, num_patches)),
            ("transpose_1", Rearrange("b d n -> b n d")),
            ("aff_post_norm", AffineTransform(dim, alpha=alpha, beta=None)),
            ("path_dropout", PathDropout(path_dropout))
        ]))
        self.channel_mixer = LayerScaleBlock(
            dim, 
            core_block=MLP, 
            pre_norm=AffineTransform, 
            alpha=alpha, 
            path_dropout=path_dropout,
            hidden_dim=4*dim,
            ff_dropout=ff_dropout
        )

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)

        return x


class ResMLPBackBone(ViTBase):
    def __init__(
        self, 
        image_size, 
        image_channel, 
        patch_size, 
        depth, 
        dim, 
        path_dropout=0., 
        token_dropout=0.,
        ff_dropout=0.,
        **kwargs
    ):
        super().__init__(image_size, image_channel, patch_size)

        self.linear_proj = nn.Linear(self.patch_dim, dim, bias=False)
        self.token_dropout = TokenWiseDropout(token_dropout)

        self.encoder = nn.Sequential(OrderedDict([
            (f"layer_{idx}", ResMLPLayer(
                dim,
                num_patches=self.num_patches,
                alpha=self._get_alpha(idx),
                path_dropout=path_dropout,
                ff_dropout=ff_dropout
            ))
            for idx in range(depth)
        ]))

        self.pooler = Reduce("b n d -> b d", "mean")

    def forward(self, x):
        # Divide into flattened patches)
        x = self.patch_and_flat(x)

        # Linear projection
        x = self.linear_proj(x)
        x = self.token_dropout(x)

        # Encoder
        x = self.encoder(x)

        # Layer norm and mean pool
        return self.pooler(x)

    def _get_alpha(self, layer_idx):
        # From https://arxiv.org/abs/2103.17239, the text under equation (4)

        if layer_idx <= 18:
            return 1e-1
        elif layer_idx <= 24:
            return 1e-5
        else:
            return 1e-6

        
class ResMLPWithLinearClassifier(ResMLPBackBone):
    def __init__(self, config: ResMLPConfig = None) -> None:
        super().__init__(**config.__dict__)

        self.proj_head = nn.LazyLinear(config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)

        return self.proj_head(x)