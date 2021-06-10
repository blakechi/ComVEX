from collections import OrderedDict

import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce

from .config import MLPMixerConfig
from comvex.vit import ViTBase
from comvex.utils import MLP


class MLPMixerLayer(nn.Module):
    def __init__(self, num_tokens, num_channels, token_mlp_dim, channel_mlp_dim, ff_dropout):
        super().__init__()

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b s c -> b c s"),
            MLP(num_tokens, token_mlp_dim, ff_dropout=ff_dropout),
            Rearrange("b c s -> b s c"),
        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLP(num_channels, channel_mlp_dim, ff_dropout=ff_dropout)
        )

    def forward(self, x):
       x = self.token_mixer(x) + x

       return self.channel_mixer(x) + x


class MLPMixerBackBone(ViTBase):
    def __init__(
        self,
        image_channel,
        image_size,  # one lateral's size of a squre image
        patch_size,  # one lateral's size of a squre patch
        *,
        depth,
        token_mlp_dim,        
        channel_mlp_dim,   
        ff_dropout=0.0,
        **kwargs
    ):
        super().__init__(image_size, image_channel, patch_size)

        self.linear_proj = nn.Linear(self.patch_dim, self.patch_dim)
        self.encoder = nn.Sequential(OrderedDict([
            (f"layer_{idx}", MLPMixerLayer(self.num_patches, self.patch_dim, token_mlp_dim, channel_mlp_dim, ff_dropout))
            for idx in range(depth)
        ]))
        self.pooler = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            Reduce("b s c -> b c", "mean")    
        )
        
    def forward(self, x):
        # Divide into flattened patches)
        x = self.patch_and_flat(x)

        # Linear projection
        x = self.linear_proj(x)

        # Encoder
        x = self.encoder(x)

        # Layer norm and mean pool
        return self.pooler(x)
        

class MLPMixerWithLinearClassifier(MLPMixerBackBone):
    def __init__(self, config: MLPMixerConfig = None) -> None:
        super().__init__(**config.__dict__)

        self.proj_head = nn.Linear(self.patch_dim, config.num_classes)
        nn.init.zeros_(self.proj_head.weight.data)  # from the original paper (Appendix E) (?)
        nn.init.zeros_(self.proj_head.bias.data)  # from the original paper (Appendix E) (?)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)

        return self.proj_head(x)