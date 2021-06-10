from comvex.utils.dropout import TokenDropout
import torch
from torch import nn, einsum
from einops import rearrange, repeat

from comvex.vit import ViTBase
from comvex.transformer import TransformerEncoderLayer
from comvex.utils import Residual, LayerNorm, FeedForward, ProjectionHead, TokenWiseDropout
from comvex.convit.config import ConViTConfig


class GatedPositionalSelfAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads=None, 
        head_dim=None,
        locality_strength=None,
        num_patches=None,
        attention_dropout=0.0, 
        dtype=torch.float32,
        **rest
    ):
        super().__init__()

        assert (
            heads is not None or head_dim is not None
        ), f"[{self.__class__.__name__}] Either `heads` or `head_dim` must be specified."

        self.heads = heads if heads is not None else dim // head_dim
        head_dim = head_dim if head_dim is not None else dim // heads

        assert (
            head_dim * heads == dim
        ), f"[{self.__class__.__name__}] Head dimension times the number of heads must be equal to embedding dimension `dim`"

        self.Q = nn.Linear(dim, dim, bias=False)
        self.K = nn.Linear(dim, dim, bias=False)
        self.V = nn.Linear(dim, dim, bias=False)
        self.postion_proj = nn.Linear(3, self.heads)
        self.out_linear = nn.Linear(dim, dim)
        self.gates = nn.Parameter(torch.ones(1, heads, 1, 1), requires_grad=True)
        self.register_buffer("relative_position_index", self._get_relative_position_index(num_patches))

        self.attention_dropout = nn.Dropout2d(attention_dropout)
        self.scale = head_dim**(-0.5)

        self.apply(self._init_weights)
        if locality_strength:
            self._init_position_related(locality_strength)

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads

        q, k, v = map(lambda proj: proj(x), (self.Q, self.K, self.V))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        patch_similarity = einsum("b h n d, b h m d -> b h n m", q, k)*self.scale  # m=n
        patch_similarity = patch_similarity.softmax(dim=-1)

        position_similarity = self.postion_proj(self.relative_position_index)
        position_similarity = rearrange(position_similarity, "1 n m h -> 1 h n m").softmax(dim=-1)

        gates = torch.sigmoid(self.gates)
        similarity = (1.0 - gates)*patch_similarity + gates*position_similarity
        similarity = self.attention_dropout(similarity)  # Since `nn.Dropout2D` rescales tensors, we don't re-normalize `similarity` along the last dimension here.

        weighted_tokens = einsum("b h n m, b h m d -> b h n d", similarity, v)
        weighted_tokens = rearrange(weighted_tokens, "b h n d -> b n (h d)")

        return self.out_linear(weighted_tokens)

    def _init_position_related(self, locality_strength):
        # Reference from: https://github.com/facebookresearch/convit/blob/main/convit.py#L124
        nn.init.eye_(self.V.weight)

        pixel_distance = 1.0
        kernel_size = int(self.heads**0.5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2
        for col in range(kernel_size):
            for row in range(kernel_size):
                position = col + kernel_size*row
                self.postion_proj.weight.data[position, 2] = -1
                self.postion_proj.weight.data[position, 1] = 2*(col - center)*pixel_distance
                self.postion_proj.weight.data[position, 0] = 2*(row - center)*pixel_distance

        self.postion_proj.weight.data *= locality_strength

    def _get_relative_position_index(self, num_patches):
        """
        Reference from: https://github.com/facebookresearch/convit/blob/main/convit.py#L139

        When `num_patches_lateral` = 5,

        idx = tensor([[ 0,  1,  2,  3,  4],
                      [-1,  0,  1,  2,  3],
                      [-2, -1,  0,  1,  2],
                      [-3, -2, -1,  0,  1],
                      [-4, -3, -2, -1,  0]])
        """

        num_patches_lateral = int(num_patches**.5)
        relative_pos_idx = torch.zeros(num_patches, num_patches, 3)
        idx = torch.arange(num_patches_lateral)[None, :] - torch.arange(num_patches_lateral)[:, None]

        idx_x = idx.repeat(num_patches_lateral, num_patches_lateral)
        idx_y = idx.repeat_interleave(num_patches_lateral, dim=0).repeat_interleave(num_patches_lateral, dim=1)
        idx_sqrDist = idx_x**2 + idx_y**2
        relative_pos_idx[..., 2] = idx_sqrDist
        relative_pos_idx[..., 1] = idx_y
        relative_pos_idx[..., 0] = idx_x
        
        return rearrange(relative_pos_idx, "n m d -> 1 n m d")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None: nn.init.zeros_(m.bias)


class ConViTLayer(nn.Module):
    def __init__(self, dim, pre_norm=False, ff_dim=None, **kwargs):
        super().__init__()

        self.attention_block = LayerNorm(
            Residual(
                GatedPositionalSelfAttention(dim, **kwargs)
            ),
            dim=dim,
            use_pre_norm=pre_norm
        )
        self.ff_block = LayerNorm(
            Residual(
                FeedForward(
                    dim=dim, hidden_dim=ff_dim if ff_dim is not None else 4*dim, **kwargs
                )
            ),
            dim=dim,
            use_pre_norm=pre_norm
        )

    def forward(self, x):
        x = self.attention_block(x)

        return self.ff_block(x)


class ConViTBackbone(nn.Module):
    def __init__(
        self,
        num_local_layers,
        num_nonlocal_layers,
        num_patches,
        dim,
        heads=None,
        head_dim=None,
        locality_strength=None,
        pre_norm=False,
        ff_dim=None,                    # If not specify, ff_dim = 4*dim
        ff_dropout=0.0,
        attention_dropout=0.0,
        **kwargs
    ):
        super().__init__()

        self.local_layers = nn.Sequential(*[
            ConViTLayer(
                dim, 
                heads=heads,
                head_dim=head_dim,
                locality_strength=locality_strength,
                num_patches=num_patches, 
                attention_dropout=attention_dropout, 
                pre_norm=pre_norm, 
                ff_dim=ff_dim, 
                ff_dropout=ff_dropout
            ) for _ in range(num_local_layers)
        ])

        self.nonlocal_layers = nn.Sequential(*[
            TransformerEncoderLayer(
                dim, 
                heads=heads, 
                head_dim=head_dim, 
                attention_dropout=attention_dropout, 
                pre_norm=pre_norm, 
                ff_dim=ff_dim, 
                ff_dropout=ff_dropout
            ) for _ in range(num_nonlocal_layers)
        ])

    def forward(self, x, cls_token):
        b, n, d = x.shape

        x = self.local_layers(x)

        # Prepend CLS token and add position code
        cls_token = repeat(cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat([cls_token, x], dim=1)

        return self.nonlocal_layers(x)


class ConViTWithLinearClassifier(ViTBase):
    def __init__(self, config: ConViTConfig = None):
        super().__init__(config.image_size, config.image_channel, config.patch_size)

        self.linear_proj = nn.Linear(self.patch_dim, config.dim, bias=False)
        self.CLS = nn.Parameter(torch.randn(1, 1, config.dim), requires_grad=True)
        self.token_dropout = TokenDropout(config.token_dropout)
        # self.token_dropout = TokenWiseDropout(config.token_dropout)

        self.backbone = ConViTBackbone(num_patches=self.num_patches, **config.__dict__)

        self.proj_head = ProjectionHead(
            config.dim,
            config.num_classes,
            config.pred_act_fnc_name,
        )

    def forward(self, x):
        b, _, _, _ = x.shape  # b, c, h, w = x.shape

        x = self.patch_and_flat(x)
        x = self.linear_proj(x)
        x = self.token_dropout(x)

        x = self.backbone(x, self.CLS)

        return self.proj_head(x[:, 0, :])