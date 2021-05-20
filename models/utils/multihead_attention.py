import torch
from torch import nn, einsum
from einops import rearrange, repeat


class MultiheadAttention(nn.Module):
    def __init__(
        self, 
        in_dim, 
        heads, 
        *,
        kv_dim=None, 
        head_dim=None, 
        proj_dim=None, 
        attention_dropout=0.0, 
        dtype=torch.float32, 
        **rest
    ):
        super().__init__()

        self.heads = heads
        dim = proj_dim if proj_dim is not None else in_dim
        head_dim = head_dim if head_dim is not None else dim // heads

        assert (
            head_dim * self.heads == dim
        ), f"[{self.__class__.__name__}] Head dimension times the number of heads must be equal to embedding dimension (`in_dim` or `proj_dim`)"
        
        self.Q = nn.Linear(in_dim, dim, bias=False)
        self.K = nn.Linear(kv_dim if kv_dim is not None else in_dim, dim, bias=False)
        self.V = nn.Linear(kv_dim if kv_dim is not None else in_dim, dim, bias=False)
        self.out_linear = nn.Linear(dim, in_dim)

        # Reference from https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertModel
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.scale = head_dim ** (-0.5)
        self.mask_value = -torch.finfo(dtype).max  # pytorch default float type

    def forward(self, x, attention_mask=None):
        """
        Args:
            x (b, n, d) or ((b, n, d), (b, n, d), (b, n, d)): input tensors, if its a list, the order represents (q, k, v)
            attention_mask (b n m): Use True or 1 to mask out attention weights and False or 0 for opposite.
        """
        
        if isinstance(x, tuple):
            b, n, d, h = *x[0].shape, self.heads
            q, k, v = map(lambda proj_token_pair: proj_token_pair[0](proj_token_pair[1]), zip((self.Q, self.K, self.V), x))
        else:
            b, n, d, h = *x.shape, self.heads
            q, k, v = map(lambda proj: proj(x), (self.Q, self.K, self.V))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        similarity = einsum("b h n d, b h m d -> b h n m", q, k)*self.scale  # m=n

        if attention_mask is not None:
            attention_mask = repeat(attention_mask, "b 1 n m -> b h n m", h=h)
            similarity.masked_fill_(attention_mask, self.mask_value)

        # attention
        similarity = similarity.softmax(dim=-1)
        similarity = self.attention_dropout(similarity)
        weighted_tokens = einsum("b h n m, b h m d -> b h n d", similarity, v)
        weighted_tokens = rearrange(weighted_tokens, "b h n d -> b n (h d)")

        return self.out_linear(weighted_tokens)