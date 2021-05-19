from collections import OrderedDict

import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce

from .config import MLPMixerConfig


class MLPMixerBase(nn.Module):
    def __init__(self, image_channel, image_size, patch_size):
        super().__init__()

        assert image_channel is not None, f"[{self.__class__.__name__}] Please specify the number of input images' channels."
        assert image_size is not None, f"[{self.__class__.__name__}] Please specify input images' size."
        assert patch_size is not None, f"[{self.__class__.__name__}] Please specify patches' size."

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = (patch_size**2) * image_channel

        assert (
            (self.num_patches**0.5) * patch_size == image_size
        ), f"[{self.__class__.__name__}] Image size must be divided by the patch size."

        self.patch_and_flat = Rearrange("b c (h p) (w q) -> b (h w) (p q c)", p=self.patch_size, q=self.patch_size)
        

class MLPMixerMLP(nn.Module):
    def __init__(self, dim, hidden_dim, ff_dropout):
        super().__init__()

        self._net = nn.Sequential(
            nn.Linear(dim, hidden_dim, ff_dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, ff_dropout)
        )

    def forward(self, x):
        return self._net(x)


class MLPMixerLayer(nn.Module):
    def __init__(self, num_tokens, num_channels, token_mlp_dim, channel_mlp_dim, ff_dropout):
        super().__init__()

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b s c -> b c s"),
            MLPMixerMLP(num_tokens, token_mlp_dim, ff_dropout),
            Rearrange("b c s -> b s c"),
        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPMixerMLP(num_channels, channel_mlp_dim, ff_dropout)
        )

    def forward(self, x):
       x = self.token_mixer(x) + x

       return self.channel_mixer(x) + x


class MLPMixerBackBone(MLPMixerBase):
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
        super().__init__(image_channel, image_size, patch_size)

        self.linear_proj = nn.Linear(self.dim, self.dim)
        self.encoder = nn.Sequential(OrderedDict([
            (f"layer_{idx}", MLPMixerLayer(self.num_patches, self.dim, token_mlp_dim, channel_mlp_dim, ff_dropout))
            for idx in range(depth)
        ]))
        self.pooler = nn.Sequential(
            nn.LayerNorm(self.dim),
            Reduce("b s c -> b () c", "mean")    
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
        

class MLPMixer(MLPMixerBackBone):
    def __init__(self, config: MLPMixerConfig = None) -> None:
        super().__init__(**config.__dict__)

        self.proj_head = nn.Linear(self.dim, config.num_classes)
        nn.init.zeros_(self.proj_head.weight.data)  # from the original paper
        nn.init.zeros_(self.proj_head.bias.data)  # from the original paper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Divide into flattened patches)
        x = self.patch_and_flat(x)

        # Linear projection
        x = self.linear_proj(x)

        # Encoder
        x = self.encoder(x)

        # Layer norm and mean pool
        x = self.pooler(x)

        return self.proj_head(x)