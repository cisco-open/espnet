"""Bounded-context blocks for the VoCodec generator."""

import math
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn

from espnet2.gan_codec.shared.encoder.seanet import ConvLayerNorm, SConv1d


class ConvNeXtBlock1d(nn.Module):
    """Streamable 1-D ConvNeXt block with a causal depthwise convolution."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 7,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.depthwise = SConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            causal=True,
            norm="none",
            pad_mode="constant",
        )
        self.norm = ConvLayerNorm(channels, eps=1e-6)
        self.pointwise_in = nn.Conv1d(channels, hidden_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.pointwise_out = nn.Conv1d(hidden_channels, channels, kernel_size=1)

        if layer_scale_init_value is None:
            self.gamma = None
        elif layer_scale_init_value <= 0:
            raise ValueError("layer_scale_init_value must be positive or None")
        else:
            self.gamma = nn.Parameter(
                torch.full((channels, 1), float(layer_scale_init_value))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise_in(x)
        x = self.activation(x)
        x = self.pointwise_out(x)
        if self.gamma is not None:
            x = x * self.gamma
        return residual + x


class CausalResNetBlock1d(nn.Module):
    """Pre-normalized causal residual block used in VoCodec's PosNet."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        groups: Tuple[int, int] = (1, 1),
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(groups) != 2 or any(channels % group != 0 for group in groups):
            raise ValueError("groups must contain two divisors of channels")
        self.norm1 = ConvLayerNorm(channels)
        self.conv1 = SConv1d(
            channels,
            channels,
            kernel_size,
            groups=groups[0],
            causal=True,
            norm="none",
            pad_mode="constant",
        )
        self.norm2 = ConvLayerNorm(channels)
        self.conv2 = SConv1d(
            channels,
            channels,
            kernel_size,
            groups=groups[1],
            causal=True,
            norm="none",
            pad_mode="constant",
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(self.activation(self.norm1(x)))
        x = self.conv2(self.dropout(self.activation(self.norm2(x))))
        return residual + x


class BoundedMultiheadAttention(nn.Module):
    """Self-attention with a finite past window and configurable lookahead.

    ``query_tile_size`` only limits temporary tensors and GPU work granularity.
    It never changes the receptive field and therefore adds no algorithmic
    latency. Queries in each tile are evaluated in parallel on the GPU.

    Rotary positional embeddings are optional. When enabled, standard absolute
    position rotations are applied to queries and keys before query tiling. The
    resulting attention scores depend on relative positions, while values are
    left unchanged.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        context_frames: int = 300,
        lookahead_frames: int = 0,
        query_tile_size: int = 64,
        bias: bool = True,
        use_rope: bool = False,
        rotary_dim: Optional[int] = None,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if context_frames < 1 or lookahead_frames < 0 or query_tile_size < 1:
            raise ValueError("invalid attention context, lookahead, or tile size")
        head_dim = channels // num_heads
        if rotary_dim is None:
            rotary_dim = head_dim - head_dim % 2
        if use_rope and (rotary_dim < 2 or rotary_dim > head_dim or rotary_dim % 2):
            raise ValueError(
                "rotary_dim must be a positive even value no larger than head_dim"
            )
        if rope_base <= 0:
            raise ValueError("rope_base must be positive")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.context_frames = context_frames
        self.lookahead_frames = lookahead_frames
        self.query_tile_size = query_tile_size
        self.use_rope = use_rope
        self.rotary_dim = rotary_dim
        self.rope_base = rope_base
        self.norm = ConvLayerNorm(channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, kernel_size=1, bias=bias)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=bias)
        if use_rope:
            inverse_frequency = 1.0 / (
                rope_base
                ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
            )
        else:
            inverse_frequency = torch.empty(0)
        # This is derived from the configuration and need not be checkpointed.
        self.register_buffer(
            "rope_inverse_frequency", inverse_frequency, persistent=False
        )

    def _project_qkv(self, x: torch.Tensor):
        batch, _, frames = x.shape
        qkv = self.qkv(self.norm(x)).view(
            batch, 3, self.num_heads, self.head_dim, frames
        )
        q, k, v = qkv.unbind(dim=1)
        return q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)

    def _apply_rope(
        self, tensor: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Apply pairwise rotary embeddings to the leading feature dimensions."""
        if not self.use_rope:
            return tensor
        angles = torch.outer(
            positions.to(self.rope_inverse_frequency.dtype),
            self.rope_inverse_frequency,
        )
        cosines = angles.cos().to(tensor.dtype)[None, None]
        sines = angles.sin().to(tensor.dtype)[None, None]
        rotary = tensor[..., : self.rotary_dim]
        even, odd = rotary[..., 0::2], rotary[..., 1::2]
        rotated = torch.stack(
            (even * cosines - odd * sines, even * sines + odd * cosines),
            dim=-1,
        ).flatten(-2)
        return torch.cat((rotated, tensor[..., self.rotary_dim :]), dim=-1)

    def _qkv(self, x: torch.Tensor, position_offset: int = 0):
        q, k, v = self._project_qkv(x)
        if self.use_rope:
            positions = torch.arange(q.shape[-2], device=q.device) + position_offset
            q = self._apply_rope(q, positions)
            k = self._apply_rope(k, positions)
        return q, k, v

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_tile_size: int,
    ) -> torch.Tensor:
        frames = q.shape[-2]
        outputs = []
        scale = 1.0 / math.sqrt(self.head_dim)
        for query_start in range(0, frames, query_tile_size):
            query_end = min(frames, query_start + query_tile_size)
            key_start = max(0, query_start - self.context_frames + 1)
            key_end = min(frames, query_end + self.lookahead_frames)

            query = q[..., query_start:query_end, :]
            key = k[..., key_start:key_end, :]
            value = v[..., key_start:key_end, :]
            scores = torch.matmul(query, key.transpose(-1, -2)) * scale

            query_positions = torch.arange(query_start, query_end, device=q.device)[
                :, None
            ]
            key_positions = torch.arange(key_start, key_end, device=q.device)[None, :]
            valid = (key_positions >= query_positions - self.context_frames + 1) & (
                key_positions <= query_positions + self.lookahead_frames
            )
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            outputs.append(torch.matmul(weights, value))
        return torch.cat(outputs, dim=-2)

    def _finish(self, x: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        batch, _, frames, _ = attended.shape
        attended = attended.transpose(-1, -2).reshape(batch, self.channels, frames)
        return x + self.out_proj(attended)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        q, k, v = self._qkv(x, position_offset=position_offset)
        return self._finish(x, self._attention(q, k, v, self.query_tile_size))

    def forward_reference(
        self, x: torch.Tensor, position_offset: int = 0
    ) -> torch.Tensor:
        """Dense reference path used to verify the tiled implementation."""
        q, k, v = self._qkv(x, position_offset=position_offset)
        return self._finish(x, self._attention(q, k, v, x.shape[-1]))


class PosNet(nn.Module):
    """Four residual blocks with bounded attention after a selected block."""

    def __init__(
        self,
        channels: int,
        num_blocks: int = 4,
        kernel_size: int = 3,
        groups: Union[Tuple[int, int], Sequence[Tuple[int, int]]] = (1, 1),
        dropout: float = 0.1,
        attention_after: int = 2,
        attention_heads: int = 4,
        attention_context_frames: int = 300,
        attention_lookahead_frames: int = 0,
        attention_query_tile_size: int = 64,
        attention_use_rope: bool = False,
        attention_rotary_dim: Optional[int] = None,
        attention_rope_base: float = 10000.0,
    ):
        super().__init__()
        if not 0 <= attention_after <= num_blocks:
            raise ValueError("attention_after must be between zero and num_blocks")
        if len(groups) == 2 and all(isinstance(group, int) for group in groups):
            block_groups = [tuple(groups)] * num_blocks
        else:
            if len(groups) != num_blocks:
                raise ValueError(
                    "groups must be one pair or one pair for each PosNet block"
                )
            block_groups = [tuple(pair) for pair in groups]
        self.blocks = nn.ModuleList(
            [
                CausalResNetBlock1d(
                    channels,
                    kernel_size=kernel_size,
                    groups=block_group,
                    dropout=dropout,
                )
                for block_group in block_groups
            ]
        )
        self.attention_after = attention_after
        self.attention = BoundedMultiheadAttention(
            channels,
            num_heads=attention_heads,
            context_frames=attention_context_frames,
            lookahead_frames=attention_lookahead_frames,
            query_tile_size=attention_query_tile_size,
            use_rope=attention_use_rope,
            rotary_dim=attention_rotary_dim,
            rope_base=attention_rope_base,
        )
        self.output_norm = ConvLayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attention_after == 0:
            x = self.attention(x)
        for index, block in enumerate(self.blocks, 1):
            x = block(x)
            if index == self.attention_after:
                x = self.attention(x)
        return self.output_norm(x)
