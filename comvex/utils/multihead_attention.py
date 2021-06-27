from typing import Optional

import torch
from torch import nn, einsum
from einops import rearrange, repeat


class MultiheadAttention(nn.Module):
    def __init__(
        self, 
        in_dim, 
        *,
        heads=None, 
        kv_dim=None, 
        head_dim=None, 
        proj_dim=None,
        out_dim=None,
        attention_dropout=0.0, 
        ff_dropout=0.0, 
        dtype=torch.float32, 
        **rest
    ):
        super().__init__()

        dim = proj_dim if proj_dim is not None else in_dim
        out_dim = out_dim if out_dim is not None else in_dim

        assert (
            heads is not None or head_dim is not None
        ), f"[{self.__class__.__name__}] Either `heads` or `head_dim` must be specified."

        self.heads = heads if heads is not None else dim // head_dim
        head_dim = head_dim if head_dim is not None else dim // heads

        assert (
            head_dim * self.heads == dim
        ), f"[{self.__class__.__name__}] Head dimension times the number of heads must be equal to embedding dimension (`in_dim` or `proj_dim`)"
        
        self.Q = nn.Linear(in_dim, dim, bias=False)
        self.K = nn.Linear(kv_dim if kv_dim is not None else in_dim, dim, bias=False)
        self.V = nn.Linear(kv_dim if kv_dim is not None else in_dim, dim, bias=False)
        self.out_linear = nn.Linear(dim, out_dim)

        # Reference from`BertSelfAttention` (https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertModel)
        self.attention_dropout = nn.Dropout2d(attention_dropout)
        self.out_dropout = nn.Dropout(ff_dropout)

        self.scale = head_dim ** (-0.5)
        self.mask_value = -torch.finfo(dtype).max  # pytorch default float type

    def forward(self, x, attention_mask=None):
        """
        Args:
            x (b, n, d) or ((b, n, d), (b, n, d), (b, n, d)): input tensors, if its a list, the order represents (q, k, v)
            attention_mask (b n m): Use True or 1 to mask out attention weights and False or 0 for opposite.
        """
        h = self.heads
        
        if isinstance(x, tuple):
            q, k, v = map(lambda proj_token_pair: proj_token_pair[0](proj_token_pair[1]), zip((self.Q, self.K, self.V), x))
        else:
            q, k, v = map(lambda proj: proj(x), (self.Q, self.K, self.V))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        k = k*self.scale

        attention = einsum("b h n d, b h m d -> b h n m", q, k)
        print(attention.shape)
        if attention_mask is not None:
            attention_mask = repeat(attention_mask, "b 1 n m -> b h n m", h=h)
            attention.masked_fill_(attention_mask, self.mask_value)

        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)
        
        out = einsum("b h n m, b h m d -> b h n d", attention, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_linear(out)
        print("out:", out.shape)
        return self.out_dropout(out)


class TalkingHeadAttention(nn.Module):
    r"""
    Talking-Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf) 
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        heads_k: int = None,
        heads_v: int = None,
        head_dim: int = None,  # Unified head dimension for query, key, and value tokens
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
        use_bias=False,
        dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim or dim // heads
        self.heads = heads
        self.heads_k = heads_k or heads
        self.heads_v = heads_v or heads

        self.Q = nn.Linear(dim, self.head_dim*self.heads_k, bias=use_bias)
        self.K = nn.Linear(dim, self.head_dim*self.heads_k, bias=use_bias)
        self.V = nn.Linear(dim, self.head_dim*self.heads_v, bias=use_bias)

        self.L = nn.Conv2d(self.heads_k, self.heads, kernel_size=1, bias=False)  # Attention Logit Projection
        self.W = nn.Conv2d(self.heads, self.heads_v, kernel_size=1, bias=False)  # Attention Weight Projection

        self.out_linear = nn.Linear(self.head_dim*self.heads_v, dim)

        self.attention_dropout = nn.Dropout2d(attention_dropout)
        self.out_dropout = nn.Dropout(ff_dropout)

        self.scale = self.head_dim ** (-0.5)
        self.mask_value = -torch.finfo(dtype).max  # pytorch default float type

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (b, n, d) or ((b, n, d), (b, n, d), (b, n, d)): input tensors, if its a list, the order represents (q, k, v)
            attention_mask (b n m): Use True or 1 to mask out attention weights and False or 0 for opposite.
        """
        h_k, h, h_v = self.heads_k, self.heads, self.heads_v
        
        if isinstance(x, tuple):
            q, k, v = x[0], x[1], x[2]
        else:
            q, k, v = x, x, x

        q, k, v = self.Q(q), self.K(k), self.V(v)
        q = rearrange(q, "b n (h d) -> b h n d", h=h_k)
        k = rearrange(k, "b n (h d) -> b h n d", h=h_k)*self.scale
        v = rearrange(v, "b n (h d) -> b h n d", h=h_v)

        attention = einsum("b h n d, b h m d -> b h n m", q, k)
        attention = self.L(attention)

        if attention_mask is not None:
            attention_mask = repeat(attention_mask, "b 1 n m -> b h n m", h=h)
            attention.masked_fill_(attention_mask, self.mask_value)

        attention = attention.softmax(dim=-1)
        attention = self.W(attention)
        attention = self.attention_dropout(attention)
        
        out = einsum("b h n m, b h m d -> b h n d", attention, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_linear(out)

        return self.out_dropout(out)
