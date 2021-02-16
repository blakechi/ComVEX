import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from models.utils.base_block import Residual, Norm, FeedForward


class MultiheadAttention(nn.Module):
    def __init__(self, *, embedding_dim, heads=4, head_dim=None):
        super(MultiheadAttention, self).__init__()

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
        b, n, d, h = *x.shape, self.heads

        q, k, v = self.QKV(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale
        similarity = einsum("b h n d, b h m d -> b h n m", q, k)  # m=n

        if att_mask:
            att_mask = repeat(att_mask, "b 1 n m -> b h n m", h=h)
            similarity.masked_fill_(~att_mask, self.mask_value)

        # attention
        similarity = similarity.softmax(dim=-1)
        weighted_tokens = einsum("b h n m, b h m d -> b h n d", similarity, v)
        weighted_tokens = rearrange(weighted_tokens, "b h n d -> b n (h d)")

        return self.out_linear(weighted_tokens)


class Transformer(nn.Module):
    def __init__(
        self, *, dim, heads=4, depth, head_dim=None, ff_dim, ff_dropout=0.0, max_seq_len=128
    ):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_seq_len = max_seq_len

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            Norm(
                                MultiheadAttention(
                                    embedding_dim=dim, heads=heads, head_dim=head_dim
                                ),
                                dim=(dim),
                            ),
                        ),
                        Residual(
                            Norm(
                                FeedForward(
                                    dim=dim, hidden_dim=ff_dim, dropout=ff_dropout
                                ),
                                dim=(dim),
                            ),
                        ),
                    ]
                )
            )

    def forward(self, x, att_mask=None, padding_mask=None):
        # device = x.device

        if padding_mask is not None:
            """
            att_mask / padding_mask:

            True: Ignore
            False: To mask
            """
            att_mask &= rearrange(
                repeat(
                    padding_mask[:, :, None], "b n 1 -> b n m", m=att_mask.shape[-1]
                ),
                "b n m -> b 1 n m",
            )

        for (attn, ff) in self.layers:
            x = attn(
                x,
                att_mask=att_mask,
            )
            x = ff(x)

        return x