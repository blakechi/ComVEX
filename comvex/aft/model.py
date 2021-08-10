from comvex.utils.dropout import PathDropout
from functools import partial
from typing import Optional, Literal, Dict

import torch
from torch import nn, einsum
try:
    from typing_extensions import Final
except:
    from torch.jit import Final
    
from einops import rearrange, repeat

from comvex.vit import ViTBase
from comvex.cait import ClassAttentionLayer
from comvex.utils import MLP, ProjectionHead
from comvex.utils.helpers.functions import get_act_fnc, name_with_msg, config_pop_argument

from .config import AFTConfig


class AFTGeneral(nn.Module):
    r"""Attention Free Transformer - General

    A self-defined general module that covers AFT - Full, Simple, and Local.
    
    - Follow the rearranged form in Eq. 3 instead of Eq. 2 for the consistency with other papers.
    - Please "trace" this module to get rid of if-else statements.
    - The `Local` mode (below) isn't optimized.
    """

    use_position_bias: Final[bool]

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        position_bias_dim: int = 128,
        hidden_dim: Optional[int] = None,
        local_window_size: Optional[int] = 0,  # make sure the assert raises when not be specified
        query_act_fnc_name: str = "Sigmoid",
        use_bias: bool = False,
        use_position_bias: bool = True,
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or dim
        if local_window_size is not None:
            assert (
                (0 < local_window_size) and (local_window_size <= max_seq_len)
            ), name_with_msg(self, f"`local_window_size` should be in the interval (0, `max_seq_len`]: (0, {max_seq_len}]. But got: {local_window_size}.")

        use_local = True if local_window_size is not None else False
        self.use_position_bias = use_position_bias

        self.Q = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.K = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.V = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.out_linear = nn.Linear(hidden_dim, dim)
        
        if self.use_position_bias:
            self.u = nn.Parameter(torch.rand(max_seq_len, position_bias_dim), requires_grad=True)
            self.v = nn.Parameter(torch.rand(max_seq_len, position_bias_dim), requires_grad=True)
        
        self.query_act_fnc = get_act_fnc(query_act_fnc_name)()
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_dropout = nn.Dropout(ff_dropout)

        self.position_bias_mask = self.get_local_window_mask(local_window_size, max_seq_len) if use_local else None

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, n, _ = x.shape

        # project
        q, k, v = self.Q(x), self.K(x), self.V(x)
        
        # attention (although it's attention-free, use `attention` for consistency with other papers)
        q = self.query_act_fnc(q)
        if self.use_position_bias:
            position_bias = einsum("n d, m d -> n m", self.u, self.v)
            if self.position_bias_mask is not None:  # if local
                position_bias.mask_fill_(self.position_bias_mask, 0)

            position_bias = rearrange(position_bias, "n m -> 1 n m 1")
            k = repeat(k, "b m d -> b n m d", n=n)
            k = k + position_bias

        k = k.softmax(dim=-2)

        if self.use_position_bias:
            attention = einsum("b n d, b n m d -> b n m d", q, k)
        else:
            attention = q*k

        attention = self.attention_dropout(attention)

        # 
        if self.use_position_bias:
            out = einsum("b n m d, b m d -> b n d", attention, v)
        else:
            out = attention*v

        out = self.out_linear(out)
        out = self.out_dropout(out)

        return out

    @torch.jit.ignore
    def _init_weights(self) -> None:
        if self.use_position_bias:
            nn.init.normal_(self.u, std=0.01)
            nn.init.normal_(self.v, std=0.01)

        # other inits...
        
    @staticmethod
    @torch.jit.ignore
    def get_local_window_mask(local_window_size: int, max_seq_len: int) -> torch.Tensor:
        mask = torch.zeros(max_seq_len, max_seq_len, dtype=torch.bool)
        
        for idx in range(max_seq_len):
            begin = idx - local_window_size
            end = idx + local_window_size

            begin = begin if begin >= 0 else 0
            end = end if end <= max_seq_len else max_seq_len
            mask[idx, begin: end] = True

        return ~mask  # filter out elements out of the local window


AFTFull = partial(AFTGeneral, local_window_size=None, use_position_bias=True)


AFTSimple = partial(AFTGeneral, local_window_size=None, use_position_bias=False)


AFTLocal = partial(AFTGeneral, use_position_bias=True)


class AFTDepthWiseConv2DOperator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        dim = x.shape[-3]
        padding = weight.shape[-1] // 2

        weight = weight.unsqueeze(dim=0).expand(dim, -1, -1, -1)
        x = nn.functional.conv2d(x, weight, padding=padding, groups=dim)

        return x


class AFTConv(nn.Module):
    r"""Attention Free Transformer - Conv
    How to get K? Assume by projecting
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        local_window_size: int,
        max_seq_len: int,
        hidden_dim: Optional[int] = None,
        query_act_fnc_name: str = "Sigmoid",
        use_bias: bool = False,
        epsilon: float = 1e-8,
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or dim
        assert (
            hidden_dim % heads == 0
        ), name_with_msg(self, f"")
        
        max_seq_len = max_seq_len**0.5  # due to H and W
        assert (
            (0 < local_window_size) and (local_window_size <= max_seq_len)
        ), name_with_msg(self, f"`local_window_size` should be in the interval (0, `max_seq_len`]: (0, {max_seq_len}]. But got: {local_window_size}.")

        self.heads = heads

        self.Q = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.K = nn.Linear(dim, heads, bias=use_bias)  # not sure
        self.V = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.out_linear = nn.Linear(hidden_dim, dim)
        
        self.w = nn.Parameter(torch.rand(heads, 1, local_window_size, local_window_size), requires_grad=True)
        self.w_norm = nn.BatchNorm2d(1, eps=epsilon)
        self.conv2d = AFTDepthWiseConv2DOperator()

        self.query_act_fnc = get_act_fnc(query_act_fnc_name)()
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_dropout = nn.Dropout(ff_dropout)
        self.epsilon = epsilon

        self._init_weights()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # b, n, d = x

        #
        q, k, v = self.Q(x), self.K(x), self.V(x)
        q = rearrange(q, "b (h w) (p d) -> b p d h w", h=H, w=W, p=self.heads)
        k = rearrange(k, "b (h w) p -> b p h w", h=H, w=W)
        v = rearrange(v, "b (h w) (p d) -> b p d h w", h=H, w=W, p=self.heads)
        
        #
        q = self.query_act_fnc(q)
        k_exp = rearrange(k, "b p h w -> b p 1 h w").exp()
        global_connectivity = torch.sum(k_exp*v, dim=(-1, -2), keepdim=True)
        global_connectivity_norm = torch.sum(k_exp, dim=(-1, -2), keepdim=True)
        k_list = torch.split(k, 1, dim=1)
        v_list = torch.split(v, 1, dim=1)
        w = self.w_norm(self.w)
        
        outs = []
        for head_idx in range(self.heads):
            k_ = k_list[head_idx]
            v_ = rearrange(v_list[head_idx], "b 1 d h w -> b d h w")
            w_ = w[head_idx].exp() - 1
            numenator = self.conv2d(k_*v_, w_) + global_connectivity[:, head_idx, ...]
            denominator = self.conv2d(k_, w_) + global_connectivity_norm[:, head_idx, ...]
            out = numenator / (denominator + self.epsilon)
            print(k_.shape, v_.shape, numenator.shape, denominator.shape, self.conv2d(k_*v_, w_).shape, w_.shape)

            outs.append(out)

        outs = torch.stack(outs, dim=1)
        outs = self.attention_dropout(outs)  # temperary implementation

        #
        outs = q*outs
        outs = rearrange(outs, "b p d h w -> b (h w) (p d)")
        outs = self.out_linear(outs)
        outs = self.out_dropout(outs)

        return outs

    @torch.jit.ignore
    def _init_weights(self) -> None:
        nn.init.zeros_(self.w_norm.weight)


class AFTLayer(nn.Module):
    r"""Attention Free Transformer Layer for AFT - Full, Simple, Local, Conv or General
    """
    
    MODE_MODULE_MAP: Final[Dict[str, nn.Module]] = {
        "full": AFTFull,
        "simple": AFTSimple,
        "local": AFTLocal,
        "conv": AFTConv,
        "general": AFTGeneral,
    }

    def __init__(
        self,
        dim: int,
        mode: Literal["full", "simple", "local", "conv", "general"] = "full",
        ff_expand_scale: int = 4,
        ff_act_fnc_name: str = "GELU",
        ff_dropout: float = 0.,
        path_dropout: float = 0.,
        **kwargs
    ) -> None:
        super().__init__()
        
        self.use_conv = True if mode == "conv" else False

        self.attn_norm = nn.LayerNorm(dim)
        self.attn_block = self.MODE_MODULE_MAP[mode](
            dim=dim,
            ff_dropout=ff_dropout,
            **kwargs,
        )
        self.attn_path_drop = PathDropout(path_dropout)
        
        self.ff_norm = nn.LayerNorm(dim)
        self.ff_block = MLP(
            dim=dim,
            ff_expand_scale=ff_expand_scale,
            act_fnc_name=ff_act_fnc_name,
            ff_dropout=ff_dropout
        )
        self.ff_path_drop = PathDropout(path_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")

        if self.use_conv:
            x = x + self.attn_path_drop(self.attn_block(self.attn_norm(x), H, W))
        else:
            x = x + self.attn_path_drop(self.attn_block(self.attn_norm(x)))

        x = x + self.ff_path_drop(self.ff_block(self.ff_norm(x)))

        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x


class AFTBackbone(ViTBase):
    r"""Attention Free Transformer Backbone

    - For simplicity and because the strong global interactions from AFT might mitigate the functionality of CLS tokens, 
      instead of prepending CLS tokens at the beginning, we offer mean pooling or class attention layers from CaiT.
    """

    use_mean_pooling: Final[bool]

    def __init__(
        self,
        image_size: int,
        image_channel: int,
        patch_size: int,
        num_layers: int,
        dim: int,
        local_window_size: int,
        hidden_dim: Optional[int] = None,
        aft_mode: Literal["full", "simple", "local", "conv", "general"] = "full",
        pool_mode: Literal["mean", "class"] = "mean",
        query_act_fnc_name: str = "Sigmoid",
        use_bias: bool = False,
        ff_expand_scale: int = 4,
        ff_dropout: float = 0.,
        attention_dropout: float = 0.,
        path_dropout: float = 0.,
        # AFT - General, Full, Simple, Local
        position_bias_dim: int = 128,
        use_position_bias: bool = True,
        # AFT - Conv
        heads: int = 32,
        epsilon: float = 1e-8,
        # Possible Class Attention Layer
        alpha: float = 1e-5,
        cls_attn_heads: int = 16,
    ) -> None:
        super().__init__(image_size, image_channel, patch_size, use_patch_and_flat=True)

        self.patch_proj = nn.Conv2d(image_channel, dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        
        self.layers = nn.ModuleList([
            AFTLayer(
                dim=dim,
                mode=aft_mode,
                local_window_size=local_window_size,
                max_seq_len=self.num_patches,
                hidden_dim=hidden_dim,
                query_act_fnc_name=query_act_fnc_name,
                use_bias=use_bias,
                heads=heads,
                epsilon=epsilon,
                ff_expand_scale=ff_expand_scale,
                ff_dropout=ff_dropout,
                path_dropout=path_dropout,
                attention_dropout=attention_dropout,
            ) for _ in range(num_layers)
        ]) if aft_mode == "conv" else nn.ModuleList([
            AFTLayer(
                dim=dim,
                mode=aft_mode,
                local_window_size=local_window_size,
                max_seq_len=self.num_patches,
                hidden_dim=hidden_dim,
                query_act_fnc_name=query_act_fnc_name,
                use_bias=use_bias,
                position_bias_dim=position_bias_dim,
                use_position_bias=use_position_bias,
                ff_expand_scale=ff_expand_scale,
                ff_dropout=ff_dropout,
                path_dropout=path_dropout,
                attention_dropout=attention_dropout,
            ) for _ in range(num_layers)
        ])
        
        self.use_mean_pooling = True if pool_mode == "mean" else False
        if pool_mode == "class":
            self.CLS = nn.Parameter(torch.randn(1, 1, dim))
            self.pool = nn.ModuleList([
                ClassAttentionLayer(
                    dim=dim,
                    heads=cls_attn_heads,
                    alpha=alpha,
                    ff_expand_scale=ff_expand_scale,
                    ff_dropout=ff_dropout,
                    path_dropout=path_dropout,
                    attention_dropout=attention_dropout,
                ) for _ in range(2)
            ])
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, *rest = x.shape

        x = self.patch_proj(x)

        for layer in self.layers:
            x = layer(x)

        if self.use_mean_pooling:
            out = self.pool(x)
        else:
            out = repeat(self.CLS, "1 1 d -> b 1 d", b=b)
            x = rearrange(x, "b c h w -> b (h w) c")
            for cls_layer in self.pool:
                out = cls_layer(out, x)

        return out.view(b, -1)


class AFTWithLinearClassifier(AFTBackbone):
    def __init__(self, config: AFTConfig) -> None:
        num_classes = config_pop_argument(config, "num_classes")
        pred_act_fnc_name = config_pop_argument(config, "pred_act_fnc_name")
        super().__init__(**config.__dict__)

        self.proj_head = ProjectionHead(
            dim=config.dim,
            out_dim=num_classes,
            act_fnc_name=pred_act_fnc_name
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        
        x = super().forward(x)

        return self.proj_head(x)