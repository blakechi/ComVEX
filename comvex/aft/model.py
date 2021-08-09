from functools import partial
from typing import Optional

import torch
from torch import nn, einsum
try:
    from typing_extensions import Final
except:
    from torch.jit import Final
    
from einops import rearrange, repeat

from comvex.utils.helpers.functions import get_act_fnc, name_with_msg


class AFTGeneral(nn.Module):
    r"""Attention Free Transformer

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
        hidden_dim: Optional[int] = None,
        local_window_size: Optional[int] = 0,  # make sure the assert raises when not be specified
        query_act_fnc: str = "Sigmoid",
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
            ), name_with_msg(f"`local_window_size` should be in the interval (0, `max_seq_len`]: (0, {max_seq_len}]. But got: {local_window_size}.")

        use_local = True if local_window_size is None or local_window_size == max_seq_len else False
        self.use_position_bias = use_position_bias

        self.Q = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.K = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.V = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.out_linear = nn.Linear(hidden_dim, dim)
        
        if self.use_position_bias:
            self.u = nn.Parameter(torch.rand(max_seq_len), requires_grad=True)
            self.v = nn.Parameter(torch.rand(max_seq_len), requires_grad=True)
        
        self.query_act_fnc = get_act_fnc(query_act_fnc)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_dropout = nn.Dropout(ff_dropout)

        self.position_bias_mask = self.get_local_window_mask(local_window_size, max_seq_len) if use_local else None

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, n, _ = x

        # project
        q, k, v = self.Q(x), self.K(x), self.V(x)
        
        # attention (although it's attention-free, use `attention` for consistency with other papers)
        q = self.query_act_fnc(q)
        if self.use_position_bias:
            position_bias = einsum("n, m -> n m", self.u, self.v)
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

    def _init_weights(self):
        if self.use_position_bias:
            nn.init.normal_(self.u, std=0.01)
            nn.init.normal_(self.v, std=0.01)

        # other inits...
        
    @staticmethod
    def get_local_window_mask(local_window_size, max_seq_len):
        mask = torch.zeros(max_seq_len, max_seq_len, dtype=torch.bool)
        
        for idx in range(max_seq_len):
            begin = idx - local_window_size
            end = idx + local_window_size

            begin = begin if begin >= 0 else 0
            end = end if end <= max_seq_len else max_seq_len
            mask[idx, begin: end] = True

        return ~mask  # filter out elements out of the local window


AFTFull = partial(AFTGeneral, local_window_size=None, use_position_bias=True)


AFTSimple = partial(AFTGeneral, use_position_bias=False)


AFTLocal = partial(AFTGeneral, use_position_bias=True)


# TODO
class AFTOptimizedLocal(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()


#TODO
class SeperableConv2DOperator(nn.Module):
    def __init__(
        self,
        stride: int,
        padding: int,
        groups: int, 
        dilation: int = 1,
    ):
        super().__init__()


# TODO
class AFTConv(nn.Module):
    r"""
    Not clear from the preprint paper. How to get K?
    """
    def __init__(
        self,
        
    ) -> None:
        super().__init__()