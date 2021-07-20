from functools import partial
from typing import Optional, Literal, List

import torch
from torch import nn
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from comvex.utils.helpers.functions import get_act_fnc, get_conv_layer, get_norm_layer, name_with_msg
from .dropout import PathDropout
from .convolution import XXXConvXdBase


class Residual(nn.Module):
    def __init__(self, fn, path_dropout=0.):
        super(Residual, self).__init__()

        assert fn is not None, f"[{self.__class__.__name__}] Must give it a function (normaly, a neural net)"
        self._fn = fn
        self.path_dropout = nn.Dropout(path_dropout)
        # self.path_dropout = PathDropout(path_dropout)

    def forward(self, x, *args, **kwargs):
        if isinstance(x, tuple) or isinstance(x, list):
            return x[0] + self.path_dropout(self._fn(x, *args, **kwargs))  # assume the first element is the main hidden state
        else:
            return x + self.path_dropout(self._fn(x, *args, **kwargs))


# TODO: Refine it!!
class LayerNorm(nn.Module):
    def __init__(self, fn=None, *, dim=None, use_pre_norm=False, use_cross_attention=False, cross_dim=None):
        super(LayerNorm, self).__init__()

        self._fn = fn
        self._use_pre_norm = use_pre_norm
        self._use_cross_attention = use_cross_attention
        if self._fn:
            self._norm = nn.LayerNorm(dim)
            if self._use_pre_norm and use_cross_attention:
                assert cross_dim is not None, f"[{self.__class__.__name__}] Please specify `cross_dim` when using cross attention"
                self._cross_norm_k = nn.LayerNorm(cross_dim)
                self._cross_norm_v = nn.LayerNorm(cross_dim)

    def forward(self, x, *args, **kwargs):
        if self._fn:
            if self._use_pre_norm:
                if self._use_cross_attention:
                    x = self._norm(x[0]), self._cross_norm_k(x[1]), self._cross_norm_v(x[2])
                    return self._fn(x, *args, **kwargs)
                return self._fn(self._norm(x), *args, **kwargs)
            else:
                return self._norm(self._fn(x, *args, **kwargs))

        return self._norm(x)


class MaskLayerNorm(LayerNorm):
    """
    Args:
        x (b, n, d): input tensor
        norm_mask (b, n) (Bool Tensor): True => to 0, False => ignore 
    """

    def forward(self, x, norm_mask=None, *args, **kwargs):
        if self._fn:
            if self._use_pre_norm:
                x = self._fn(self._norm(x), *args, **kwargs)
            else:
                x = self._norm(self._fn(x, *args, **kwargs))
        else:
            x = self._norm(x)

        assert norm_mask is not None, f"[{self.__class__.__name__}] Please provide `norm_mask`."

        return x.masked_fill_(norm_mask, 0)


class FeedForward(nn.Module):
    r"""
    Feed-Forward Layer (Alias: MLP)

    Support 1x1 convolution for 1, 2, and 3D data
    """
    def __init__(
        self,
        dim: int,
        out_dim: Optional[int] = None,
        expand_dim: Optional[int] = None,
        ff_expand_scale: Optional[int] = None,
        ff_dropout: float = 0.0,
        act_fnc_name: str = "GELU",
        use_convXd: Optional[int] = None,
        **rest
    ) -> None:
        super().__init__()
        
        expand_dim = expand_dim or ff_expand_scale*dim if (expand_dim is not None) and (ff_expand_scale is not None) else dim
        out_dim = out_dim or dim

        if use_convXd:
            assert (
                0 < use_convXd and use_convXd < 4
            ), name_with_msg(f"`use_convXd` must be 1, 2, or 3 for valid `ConvXd` supported by PyTorch. But got: {use_convXd}")

            core = partial(get_conv_layer(f"Conv{use_convXd}d"), kernel_size=1)
        else:
            core = nn.Linear

        self.ff_0 = core(dim, expand_dim)
        self.act_fnc = get_act_fnc(act_fnc_name)()
        self.dropout_0 = nn.Dropout(ff_dropout)
        self.ff_1 = core(expand_dim, out_dim)
        self.dropout_1 = nn.Dropout(ff_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff_0(x)
        x = self.act_fnc(x)
        x = self.dropout_0(x)
        x = self.ff_1(x)
        x = self.dropout_1(x)

        return x

MLP = FeedForward


# Reference from: https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py#L642
class ProjectionHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, act_fnc_name: str = "ReLU", ff_dropout: float = 0.1):
        super().__init__()

        self.head = nn.Sequential(  
            nn.Linear(dim, dim),
            get_act_fnc(act_fnc_name)(),
            nn.Dropout(ff_dropout),
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x):
        return self.head(x)


class PatchEmbeddingXd(XXXConvXdBase):
    to_flat: Final[bool]
    dimension: Final[int]

    def __init__(
        self,
        image_channel: int,
        embedding_dim: int,
        patch_size: int,
        dimension: int = 2,
        to_flat: bool = True,
    ) -> None:
        super().__init__(in_channel=image_channel, out_channel=embedding_dim, dimension=dimension)

        self.proj = self.conv(
            self.in_channel,
            self.out_channel,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(self.out_channel) if to_flat else get_norm_layer(f"BatchNorm{self.dimension}d")(self.out_channel)

        self.to_flat = to_flat
        
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape

        if self.to_flat:
            if self.dimension > 1:
                x = x.flatten(-2)
            x = torch.einsum("b d n -> b n d", x)

        x = self.norm(x)

        return x, (H, W)

    @torch.jit.ignore
    def no_weight_decay(self) -> List[str]:
        return ["LayerNorm.weight"] if self.to_flat else [f"BatchNorm{self.dimension}d.weight"]