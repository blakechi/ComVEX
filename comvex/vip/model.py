from typing import Dict, List
from collections import OrderedDict

import torch
from torch import nn
from einops import rearrange, reduce

from comvex.vit import ViTBase
from comvex.utils import MLP, PathDropout, PatchEmbeddingXd, ChannelFirstLayerNorm
from comvex.utils.helpers import config_pop_argument
from .config import ViPConfig


class PermuteMLP(nn.Module):
    r"""(Weighted) Permute - MLP

    - Default is Weighted Permute - MLP, set `use_weighted` to `False` for unweighted one.
    - Reference from: https://github.com/Andrew-Qibin/VisionPermutator/blob/main/models/vip.py#L42-L78
    """
    def __init__(
        self,
        dim: int,
        use_weighted: bool = True,
        use_bias: bool = False,
        ff_dropout: float = 0.,
    ) -> None:
        super().__init__()

        self.H = nn.Conv2d(dim, dim, kernel_size=1, bias=use_bias)
        self.W = nn.Conv2d(dim, dim, kernel_size=1, bias=use_bias)
        self.C = nn.Conv2d(dim, dim, kernel_size=1, bias=use_bias)
        self.out_linear = nn.Conv2d(dim, dim, kernel_size=1)

        self.weight_proj = MLP(
            dim,
            out_dim=dim*3,
            ff_expand_scale=0.25,
            ff_dropout=ff_dropout,  # Differ from the official code
        ) if use_weighted else None

        self.out_dropout = nn.Dropout(ff_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        #
        x_h = rearrange(x, "b (p d) h w -> b (h d) p w", p=h)  # p for number of patches, simialr to heads in self-attention
        x_w = rearrange(x, "b (p d) h w -> b (w d) h p", p=w)  # p for number of patches, simialr to heads in self-attention
        
        x_h = self.H(x_h)
        x_w = self.H(x_w)
        x_c = self.C(x)

        #
        x = torch.stack([x_h, x_w, x_c], dim=0)
        if self.weight_proj is not None:
            weights = reduce(x.sum(dim=0), "b c h w -> b c", reduction="mean")
            weights = self.weight_proj(weights)
            weights = rearrange(weights, "b (p c) -> p b c 1 1", p=3).softmax(dim=0)
            x = x*weights

        x = x.sum(dim=0)

        #
        x = self.out_linear(x)
        x = self.out_dropout(x)

        return x
    

class Permutator(nn.Module):
    r"""Permutator
    """
    def __init__(
        self,
        dim: int,
        ff_dropout: float = 0.,
        path_dropout: float = 0.,
        **kwargs
    ):
        super().__init__()

        self.permute_norm = ChannelFirstLayerNorm(dim)
        self.permute_block = PermuteMLP(
            dim,
            ff_dropout=ff_dropout,
            **kwargs
        )
        self.permute_path_drop = PathDropout(path_dropout)

        self.channel_norm = ChannelFirstLayerNorm(dim)
        self.channel_block = MLP(
            dim,
            ff_expand_scale=3,
            ff_dropout=ff_dropout,
            use_convXd=2,
        )
        self.channel_path_drop = PathDropout(path_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.permute_path_drop(self.permute_block(self.permute_norm(x)))
        x = x + self.channel_path_drop(self.channel_block(self.channel_norm(x)))

        return x


class ViPDownSampler(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        patch_size: int = 2,
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        self.proj = nn.Conv2d(in_dim, out_dim, patch_size, stride=patch_size, bias=use_bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ViPBackBone(ViTBase):
    def __init__(
        self,
        image_channel: int,
        image_size: int,            # one lateral's size of a squre image
        patch_size: int,            # one lateral's size of a squre patch
        layers_in_stages: List[int],
        channels_in_stages: List[int],
        use_weighted: bool = True,  # Whether to use `Weighted Permute - MLP` or `Permute - MLP`
        use_bias: bool = False,
        ff_dropout: float = 0.,
        path_dropout: float = 0.,
    ):
        super().__init__(image_size, image_channel, patch_size, use_patch_and_flat=False)

        self.patch_proj = PatchEmbeddingXd(image_channel, channels_in_stages[0], patch_size, to_flat=False)
        
        channels_in_stages.append(channels_in_stages[-1])  # for the last stage
        self.stages = nn.ModuleList([
            self._build_stage(
                idx,
                num_layers,
                to_downsample=True if channels_in_stages[idx] != channels_in_stages[idx + 1] else False,
                dim=channels_in_stages[idx],
                out_dim=channels_in_stages[idx + 1],
                use_weighted=use_weighted,
                use_bias=use_bias,
                ff_dropout=ff_dropout,
                path_dropout=path_dropout,
            ) for idx, num_layers in enumerate(layers_in_stages)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x, _ = self.patch_proj(x)

        feature_maps: Dict[str, torch.Tensor] = dict()
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            feature_maps[f"stage_{idx}"] = x

        return feature_maps

    @torch.jit.ignore
    def _build_stage(self, stage_idx, num_layers, to_downsample: bool, **kwargs) -> nn.Module:
        in_dim = kwargs["dim"]
        out_dim = kwargs.pop("out_dim")

        stage = OrderedDict([
            (
                f"stage_{stage_idx}_layer_{layer_idx}",
                Permutator(**kwargs)
            ) for layer_idx in range(num_layers)
        ])
        if to_downsample:
            stage[f"stage_{stage_idx}_down_sample"] = ViPDownSampler(in_dim, out_dim)

        return nn.Sequential(stage)


class ViPWithLinearClassifier(ViPBackBone):
    def __init__(self, config: ViPConfig = None) -> None:
        num_classes = config_pop_argument(config, "num_classes")
        super().__init__(**config.__dict__)

        self.num_stage = len(config.layers_in_stages) - 1

        out_dim = config.channels_in_stages[-1]
        self.norm = ChannelFirstLayerNorm(out_dim)
        self.proj_head = nn.Linear(out_dim, num_classes)

        nn.init.zeros_(self.proj_head.weight.data)  # from the original paper (Appendix E) (?)
        nn.init.zeros_(self.proj_head.bias.data)  # from the original paper (Appendix E) (?)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps = super().forward(x)
        x = feature_maps[f"stage_{self.num_stage}"]

        x = self.norm(x)
        x = reduce(x, "b c h w -> b c", reduction="mean")
        
        return self.proj_head(x)