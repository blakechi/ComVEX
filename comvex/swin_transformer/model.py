import torch
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from .config import SwinTransformerConfig
from comvex.utils import LayerNorm, FeedForward, ProjectionHead, TokenDropout, PathDropout


class SwinTransformerBase(nn.Module):
    def __init__(self, image_channel, image_size, patch_size, num_channels, num_layers_in_stages, **rest):
        super().__init__()

        assert image_channel is not None, f"[{self.__class__.__name__}] Please specify the number of input images' channels."
        assert image_size is not None, f"[{self.__class__.__name__}] Please specify input images' size."
        assert patch_size is not None, f"[{self.__class__.__name__}] Please specify patches' size."
        assert len(num_layers_in_stages) == 4, f"[{self.__class__.__name__}] The number of stages should be 4, but got {len(num_layers_in_stages)}"
        for num_layers in num_layers_in_stages:
            assert num_layers % 2 == 0, f"[{self.__class__.__name__}] The number of layers in stages should be an even number, but got {num_layers}"

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_channels = num_channels
        self.num_channels_after_patching = (patch_size**2) * image_channel

        assert (
            (self.num_patches**0.5) * patch_size == image_size
        ), f"[{self.__class__.__name__}] Image size must be divided by the patch size."

        self.patch_and_flat = Rearrange("b c (h p) (w q) -> b (h w) (p q c)", p=self.patch_size, q=self.patch_size)


class WindowAttentionBase(nn.Module):
    def __init__(self, *, dim, window_size, heads=None, head_dim=None, dtype=torch.float32, **not_used):
        super().__init__()

        self.dim = dim
        self.window_size = window_size

        assert (
            heads is not None or head_dim is not None
        ), f"[{self.__class__.__name__}] Please specify `heads` or `head_dim`"
        self.heads = heads if heads is not None else dim // head_dim
        self.head_dim = head_dim if head_dim is not None else dim // heads

        assert (
            self.head_dim * self.heads == self.dim
        ), f"[{self.__class__.__name__}] Head dimension times the number of heads must be equal to embedding dimension. ({self.head_dim}*{self.heads} != {self.dim})"

        self.relative_position = nn.Parameter(
            torch.randn([self.heads, (2*self.window_size[0] - 1)*(2*self.window_size[1] - 1)], requires_grad=True)
        )
        self.register_buffer("relative_position_index", self._get_relative_position_index())

        self.scale = self.head_dim ** (-0.5)
        self.mask_value = -torch.finfo(dtype).max  # pytorch default float type

    def split_into_windows(self, x):
        x = rearrange(x, "b (p h) (q w) c -> b h w (p q) c", p=self.window_size[0], q=self.window_size[1])

        return x

    def merge_windows(self, x):
        x = rearrange(x, "b h w (p q) c -> b (p h) (q w) c", p=self.window_size[0], q=self.window_size[1])

        return x

    def _get_relative_position_index(self) -> torch.Tensor:
        r"""
        Reference from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py

        Example: When `window_size` == (2, 2)
        >> out = tensor([[4, 3, 1, 0],
                         [5, 4, 2, 1],
                         [7, 6, 4, 3],
                         [8, 7, 5, 4]])
        >> out = out.view(-1)
        """

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        return rearrange(relative_position_index, "... -> (...)")

    def _init_weights(self, m):
        if isinstance(m, nn.Parameter):
            # Reference from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L110
            nn.init.trunc_normal_(m.weight, std=0.02)


class PatchMerging(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.merge = Rearrange("b (h p) (w q) c -> b h w (p q c)", p=2, q=2)
        self.proj_head = nn.Sequential(
            nn.LayerNorm(4*channel),
            nn.Linear(4*channel, 2*channel, bias=False)
        )

    def forward(self, x):
        x = self.merge(x)

        return self.proj_head(x)


class WindowAttention(WindowAttentionBase):
    def __init__(self, *, dim, window_size, attention_dropout=0.0, **kwargs):
        super().__init__(dim=dim, window_size=window_size, **kwargs)

        self.qkv = nn.Linear(dim, 3*dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_linear = nn.Linear(dim, dim)

        self.apply(self._init_weights)

    def forward(self, x, attention_mask=None):
        b, H, W, c, g = *x.shape, self.heads

        x = self.split_into_windows(x)
        q, k, v = self.qkv(x).chunk(chunks=3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b h w n (g d) -> b g h w n d", g=g), (q, k, v))

        q = q*self.scale
        similarity = einsum("b g h w n d, b g h w m d -> b g h w n m", q, k)

        relative_position_bias = self.relative_position[:, self.relative_position_index]
        relative_position_bias = rearrange(relative_position_bias, "g (n m) -> 1 g 1 1 n m", m=similarity.shape[-1])
        similarity = similarity + relative_position_bias

        if attention_mask is not None: 
            similarity.masked_fill_(attention_mask, self.mask_value)

        similarity = similarity.softmax(dim=-1) 
        similarity = self.attention_dropout(similarity)

        out = einsum("b g h w n m, b g h w m d -> b g h w n d", similarity, v)
        out = rearrange(out, "b g h w n d -> b h w n (g d)")
        out = self.merge_windows(out)

        return self.out_linear(out)


class ShiftWindowAttention(WindowAttention):
    def __init__(self, *, dim, window_size, shifts, input_resolution, **kwargs):
        super().__init__(dim=dim, window_size=window_size, **kwargs)

        self.shifts = shifts
        self.register_buffer("shifted_attention_mask", self._get_shifted_attnetion_mask(input_resolution))

    def forward(self, x, attention_mask=None):
        # b, H, W, c = x.shape

        x = torch.roll(x, (-self.shifts, -self.shifts), (-3, -2))

        attention_mask = attention_mask | self.shifted_attnetion_mask if attention_mask is not None else self.shifted_attention_mask
        x = super().forward(x, attention_mask)

        x = torch.roll(x, (self.shifts, self.shifts), (-3, -2))

        return x

    def _get_shifted_attnetion_mask(self, input_resolution):
        # Reference from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L210

        H, W = input_resolution
        image_mask = torch.zeros([1, H, W, 1])
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shifts),
                    slice(-self.shifts, None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shifts),
                    slice(-self.shifts, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                image_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.split_into_windows(image_mask)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask != 0  # (num_windows*num_window, window_size[0]**2, window_size[1]**2)

        return rearrange(attn_mask, "(h w) n m -> 1 1 h w n m", h=int(attn_mask.shape[0]**0.5))


class SwinTransformerLayer(nn.Module):
    def __init__(
        self, 
        dim, 
        window_size, 
        shifts=None, 
        input_resolution=None,
        ff_dim=None, 
        use_pre_norm=False, 
        **kwargs
    ):
        super().__init__()

        self.attention_block = LayerNorm(
            ShiftWindowAttention(
                dim=dim,
                window_size=window_size,
                shifts=shifts,
                input_resolution=input_resolution,
                **kwargs
            ) if shifts is not None else WindowAttention(
                dim=dim,
                window_size=window_size,
                **kwargs
            ),
            dim=dim,
            use_pre_norm=use_pre_norm
        )
        self.attention_path_dropout = PathDropout(kwargs["path_dropout"] if "path_dropout" in kwargs else 0.)

        self.ff_block = LayerNorm(
            FeedForward(
                dim=dim, hidden_dim=ff_dim if ff_dim is not None else 4*dim, **kwargs
            ),
            dim=dim,
            use_pre_norm=use_pre_norm
        )
        self.ff_path_dropout = PathDropout(kwargs["path_dropout"] if "path_dropout" in kwargs else 0.)


    def forward(self, x, attention_mask):
        x = x + self.attention_path_dropout(self.attention_block(x, attention_mask))
        x = x + self.ff_path_dropout(self.ff_block(x))

        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self, 
        num_layers,
        input_channel, 
        head_dim, 
        window_size,
        shifts=None, 
        input_resolution=None,
        use_checkpoint=False,
        **kwargs
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.layers = nn.ModuleList([
            SwinTransformerLayer(
                dim=input_channel, 
                head_dim=head_dim,
                window_size=window_size, 
                shifts=None if idx % 2 == 0 else shifts, 
                input_resolution=None if idx % 2 == 0 else input_resolution, 
                **kwargs
            ) for idx in range(num_layers)
        ])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            # Reference from: https://github.com/microsoft/Swin-Transformer/blob/a011aad339e0514e538313ee4011b4f3569f70de/models/swin_transformer.py#L391
            # Note: It's different from Sparse Transformer, but we follow the official code here.
            if self.use_checkpoint:
                x = checkpoint(layer, x, attention_mask)
            else:
                x = layer(x, attention_mask)

        return x


class SwinTransformerBackbone(SwinTransformerBase):
    def __init__(
        self, 
        image_channel, 
        image_size, 
        patch_size,
        num_channels,
        num_layers_in_stages, 
        *,
        head_dim=32,
        window_size=(7, 7),
        shifts=2,
        use_absolute_position=False,
        token_dropout=0.0,
        **kwargs
    ):
        super().__init__(image_channel, image_size, patch_size, num_channels, num_layers_in_stages)

        self.patch_embedding = nn.Linear(self.num_channels_after_patching, self.num_channels)

        self.use_absolute_position = use_absolute_position
        if use_absolute_position:
            self.absolute_position = nn.Parameter(torch.empty(1, self.num_patches, self.num_channels))
            nn.init.trunc_normal_(self.absolute_position, std=0.02)

        self.token_dropout = TokenDropout(token_dropout)

        self.stages = nn.ModuleList([
            nn.ModuleList(self._build_stage(
                f"stage_{idx}", 
                num_layers_in_stages[idx], 
                head_dim, 
                window_size, 
                shifts, 
                **kwargs
            )) for idx in range(4)
        ])

        self.pooler = nn.Sequential(
            Rearrange("b h w c -> b (h w) c"),
            nn.LayerNorm(self.num_channels * 2**(3)),
            Reduce("b n c -> b c", reduction="mean")  # token-wise mean pooling
        )

    def forward(self, x, attention_mask=None):

        x = self.patch_and_flat(x)
        x = self.patch_embedding(x)

        if self.use_absolute_position: 
            x = x + self.absolute_position

        x = self.token_dropout(x)
        x = rearrange(x, "b (h w) c -> b h w c", h=int(self.num_patches**0.5))

        for patch_merge, swin_block in self.stages:
            x = patch_merge(x)
            x = swin_block(x, attention_mask)

        x = self.pooler(x)

        return x

    def _build_stage(self, name, num_layers, head_dim, window_size, shifts, **kwargs):
        stage_idx = int(name[-1])
        input_channel = self.num_channels * 2**(stage_idx)
        input_resolution = self.image_size // 2**(stage_idx + 2)
        assert (
            (input_resolution % window_size[0] == 0) and (input_resolution % window_size[1] == 0)
        ), f"[{self.__class__.__name__}] Input resolution ({input_resolution}) of Stage {stage_idx} can not be divided by window size {window_size}."
        input_resolution = (input_resolution, input_resolution)

        block = SwinTransformerBlock(
            num_layers,
            input_channel, 
            head_dim, 
            window_size,
            shifts=shifts, 
            input_resolution=input_resolution,
            **kwargs
        )

        return nn.Identity() if stage_idx == 0 else PatchMerging(input_channel//2), block


class SwinTransformerWithLinearClassifier(SwinTransformerBackbone):
    def __init__(self, config: SwinTransformerConfig = None) -> None:
        super().__init__(**config.__dict__)

        self.proj_head = ProjectionHead(
            self.num_channels * 2**(3),
            config.num_classes,
            config.pred_act_fnc_name,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = super().forward(x, attention_mask)

        return self.proj_head(x)