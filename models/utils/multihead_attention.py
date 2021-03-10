import torch
from torch import nn, einsum
from einops import rearrange, repeat


class MultiheadAttention(nn.Module):
    def __init__(self, *, embedding_dim, heads=4, head_dim=None):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.heads = heads
        self.head_dim = head_dim if head_dim is not None else embedding_dim // heads

        assert (
            self.head_dim * self.heads == self.embedding_dim
        ), "Head dimension times the number of heads must be equal to embedding dimension"

        self.QKV = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_linear = nn.Linear(embedding_dim, embedding_dim)

        self.scale = self.head_dim ** (-0.5)
        self.mask_value = -torch.finfo(torch.float32).max  # pytorch default float type

    def forward(self, x, att_mask=None):
        """
        Args:
            x (b, n, d): input tensors
            att_mask (b n m): Use True or 1 to mask out attention weights and False or 0 for opposite.
        """
        b, n, d, h = *x.shape, self.heads

        q, k, v = self.QKV(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale
        similarity = einsum("b h n d, b h m d -> b h n m", q, k)  # m=n

        if att_mask is not None:
            att_mask = repeat(att_mask, "b 1 n m -> b h n m", h=h)
            similarity.masked_fill_(att_mask, self.mask_value)

        # attention
        similarity = similarity.softmax(dim=-1)
        weighted_tokens = einsum("b h n m, b h m d -> b h n d", similarity, v)
        weighted_tokens = rearrange(weighted_tokens, "b h n d -> b n (h d)")

        return self.out_linear(weighted_tokens)