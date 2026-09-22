# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

# MIT License
#
# Copyright (c) 2023 Descript, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""DAC-style trainable residual vector quantization.

Adapted from Descript Audio Codec's quantizer implementation:
https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/quantize.py

This version adds explicit projection modes, normalized lookup, deterministic
inference, and per-example dropout masks while retaining DAC's stop-gradient
directions for codebook and commitment losses.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from espnet2.gan_codec.shared.encoder.seanet import apply_parametrization_norm


def _projection(
    in_channels: int,
    out_channels: int,
    bias: bool,
    norm: str,
) -> nn.Module:
    if in_channels == out_channels:
        return nn.Identity()
    if norm not in ("none", "weight_norm"):
        raise ValueError("projection_norm must be 'none' or 'weight_norm'")
    layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
    return apply_parametrization_norm(layer, norm)


@dataclass
class DACQuantizedResult:
    """Outputs of :class:`DACResidualVectorQuantizer`."""

    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    num_quantizers: torch.Tensor


class DACVectorQuantization(nn.Module):
    """One trainable DAC vector-quantization layer.

    Nearest-neighbor selection may use normalized encoder and codebook vectors.
    The selected raw embedding is used for reconstruction. End-to-end losses do
    not update the embedding because of the straight-through estimator; the
    embedding is trained by ``codebook_loss`` only.
    """

    def __init__(
        self,
        input_dim: int,
        codebook_dim: int = 8,
        codebook_size: int = 1024,
        projection_bias: bool = True,
        projection_norm: str = "weight_norm",
        normalized_lookup: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.normalized_lookup = normalized_lookup
        self.in_proj = _projection(
            input_dim, codebook_dim, projection_bias, projection_norm
        )
        self.out_proj = _projection(
            codebook_dim, input_dim, projection_bias, projection_norm
        )
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(
            self.codebook.weight,
            -1.0 / math.sqrt(codebook_dim),
            1.0 / math.sqrt(codebook_dim),
        )

    def _lookup(self, z_e: torch.Tensor) -> torch.Tensor:
        vectors = z_e.transpose(1, 2)
        codebook = self.codebook.weight
        if self.normalized_lookup:
            vectors = F.normalize(vectors, dim=-1)
            codebook = F.normalize(codebook, dim=-1)
        distance = (
            vectors.pow(2).sum(dim=-1, keepdim=True)
            - 2 * torch.einsum("btd,kd->btk", vectors, codebook)
            + codebook.pow(2).sum(dim=-1)
        )
        return distance.argmin(dim=-1)

    def decode_code(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode indices shaped ``(batch, frames)`` into model channels."""
        z_q = F.embedding(codes, self.codebook.weight).transpose(1, 2)
        return self.out_proj(z_q)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.in_proj(z)
        codes = self._lookup(z_e)
        z_q = F.embedding(codes, self.codebook.weight).transpose(1, 2)

        commitment_loss = (z_e - z_q.detach()).pow(2).mean(dim=(1, 2))
        codebook_loss = (z_q - z_e.detach()).pow(2).mean(dim=(1, 2))

        z_q = z_e + (z_q - z_e).detach()
        return self.out_proj(z_q), commitment_loss, codebook_loss, codes


class DACResidualVectorQuantization(nn.Module):
    """A DAC RVQ stack with configurable projections and per-example masks.

    Projection modes:

    * ``per_layer``: each RVQ layer projects model channels to/from codebook
      channels, matching the official DAC organization.
    * ``shared``: project once before and once after the complete RVQ stack.
    * ``none``: quantize directly; model and codebook dimensions must match.
    """

    def __init__(
        self,
        input_dim: int,
        num_quantizers: int,
        codebook_dim: int = 8,
        codebook_size: int = 1024,
        projection_mode: str = "per_layer",
        projection_bias: bool = True,
        projection_norm: str = "weight_norm",
        normalized_lookup: bool = True,
        loss_normalization: str = "full_batch",
    ):
        super().__init__()
        if projection_mode not in ("per_layer", "shared", "none"):
            raise ValueError("projection_mode must be 'per_layer', 'shared', or 'none'")
        if loss_normalization not in ("full_batch", "active_examples"):
            raise ValueError(
                "loss_normalization must be 'full_batch' or 'active_examples'"
            )
        if projection_mode == "none" and input_dim != codebook_dim:
            raise ValueError(
                "projection_mode='none' requires input_dim == codebook_dim"
            )

        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self.codebook_dim = codebook_dim
        self.projection_mode = projection_mode
        self.loss_normalization = loss_normalization

        if projection_mode == "shared":
            self.in_proj = _projection(
                input_dim, codebook_dim, projection_bias, projection_norm
            )
            self.out_proj = _projection(
                codebook_dim, input_dim, projection_bias, projection_norm
            )
            layer_dim = codebook_dim
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
            layer_dim = input_dim

        layer_codebook_dim = (
            codebook_dim if projection_mode == "per_layer" else layer_dim
        )
        self.layers = nn.ModuleList(
            [
                DACVectorQuantization(
                    input_dim=layer_dim,
                    codebook_dim=layer_codebook_dim,
                    codebook_size=codebook_size,
                    projection_bias=projection_bias,
                    projection_norm=projection_norm,
                    normalized_lookup=normalized_lookup,
                )
                for _ in range(num_quantizers)
            ]
        )

    def _reduce_loss(self, loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = loss * mask
        if self.loss_normalization == "active_examples":
            return masked.sum() / mask.sum().clamp_min(1.0)
        return masked.mean()

    def forward(
        self,
        z: torch.Tensor,
        num_quantizers: Union[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = z.shape[0]
        if isinstance(num_quantizers, int):
            active_counts = torch.full(
                (batch_size,), num_quantizers, device=z.device, dtype=torch.long
            )
        else:
            active_counts = num_quantizers.to(device=z.device, dtype=torch.long)
            if active_counts.shape != (batch_size,):
                raise ValueError("num_quantizers tensor must have shape (batch,)")
        if torch.any(active_counts < 1) or torch.any(
            active_counts > self.num_quantizers
        ):
            raise ValueError("num_quantizers must be between 1 and the RVQ depth")

        residual = self.in_proj(z)
        quantized = torch.zeros_like(residual)
        all_codes: List[torch.Tensor] = []
        commitment_loss = z.new_zeros(())
        codebook_loss = z.new_zeros(())

        # Keep every layer in the training graph, including layers that happen to
        # be inactive for the whole local batch. Evaluation can skip the unused
        # suffix because it does not need DDP graph consistency.
        layer_count = (
            self.num_quantizers if self.training else int(active_counts.max().item())
        )
        for index, layer in enumerate(self.layers[:layer_count]):
            layer_quantized, layer_commit, layer_codebook, codes = layer(residual)
            mask = (active_counts > index).to(z.dtype)
            expanded_mask = mask[:, None, None]
            # Match DAC's residual path: later quantizers also backpropagate
            # through the earlier projected residuals.
            residual = residual - layer_quantized
            quantized = quantized + layer_quantized * expanded_mask
            commitment_loss = commitment_loss + self._reduce_loss(layer_commit, mask)
            codebook_loss = codebook_loss + self._reduce_loss(layer_codebook, mask)
            all_codes.append(codes.masked_fill(mask[:, None] == 0, -1))

        return (
            self.out_proj(quantized),
            torch.stack(all_codes),
            commitment_loss,
            codebook_loss,
        )

    def encode(self, z: torch.Tensor, num_quantizers: int) -> torch.Tensor:
        """Encode a deterministic RVQ prefix, returning ``(layers, batch, time)``."""
        if not 1 <= num_quantizers <= self.num_quantizers:
            raise ValueError("num_quantizers must be between 1 and the RVQ depth")
        residual = self.in_proj(z)
        all_codes = []
        for layer in self.layers[:num_quantizers]:
            z_e = layer.in_proj(residual)
            codes = layer._lookup(z_e)
            layer_quantized = layer.decode_code(codes)
            residual = residual - layer_quantized
            all_codes.append(codes)
        return torch.stack(all_codes)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes shaped ``(layers, batch, time)``."""
        if codes.ndim != 3:
            raise ValueError("codes must have shape (layers, batch, time)")
        if not 1 <= codes.shape[0] <= self.num_quantizers:
            raise ValueError("code layer count is outside the configured RVQ depth")
        quantized = None
        for layer, layer_codes in zip(self.layers, codes):
            layer_quantized = layer.decode_code(layer_codes)
            quantized = (
                layer_quantized if quantized is None else quantized + layer_quantized
            )
        return self.out_proj(quantized)


class DACResidualVectorQuantizer(nn.Module):
    """Bandwidth-aware wrapper around :class:`DACResidualVectorQuantization`.

    During training, if ``num_quantizers`` is omitted, a fraction
    ``quantizer_dropout`` of examples uses a uniformly sampled RVQ prefix from
    ``dropout_num_quantizers``. The remaining examples use all layers. If no
    candidates are supplied, every prefix from one through ``n_q`` is eligible
    for backward compatibility. An explicitly supplied prefix is always
    respected. Evaluation never samples and defaults to all layers.
    """

    def __init__(
        self,
        dimension: int = 192,
        codebook_dim: int = 8,
        n_q: int = 32,
        bins: int = 1024,
        quantizer_dropout: float = 0.5,
        projection_mode: str = "per_layer",
        projection_bias: bool = True,
        projection_norm: str = "weight_norm",
        normalized_lookup: bool = True,
        loss_normalization: str = "full_batch",
    ):
        super().__init__()
        if isinstance(quantizer_dropout, bool) or not 0.0 <= quantizer_dropout <= 1.0:
            raise ValueError("quantizer_dropout must be a float between 0 and 1")
        self.dimension = dimension
        self.codebook_dim = codebook_dim
        self.n_q = n_q
        self.bins = bins
        self.quantizer_dropout = float(quantizer_dropout)
        self.vq = DACResidualVectorQuantization(
            input_dim=dimension,
            num_quantizers=n_q,
            codebook_dim=codebook_dim,
            codebook_size=bins,
            projection_mode=projection_mode,
            projection_bias=projection_bias,
            projection_norm=projection_norm,
            normalized_lookup=normalized_lookup,
            loss_normalization=loss_normalization,
        )

    def get_bandwidth_per_quantizer(self, frame_rate: int) -> float:
        return math.log2(self.bins) * frame_rate / 1000.0

    def get_num_quantizers_for_bandwidth(
        self, frame_rate: int, bandwidth: Optional[float]
    ) -> int:
        if bandwidth is None or bandwidth <= 0:
            return self.n_q
        count = math.floor(bandwidth / self.get_bandwidth_per_quantizer(frame_rate))
        return max(1, min(self.n_q, count))

    def get_num_quantizers_for_bandwidths(
        self,
        frame_rate: int,
        bandwidths: Sequence[float],
    ) -> Tuple[int, ...]:
        """Map exactly representable bandwidths to distinct RVQ prefix lengths."""
        if isinstance(bandwidths, (str, bytes)) or not bandwidths:
            raise ValueError("dropout bandwidths must be a non-empty sequence")

        bandwidth_per_quantizer = self.get_bandwidth_per_quantizer(frame_rate)
        counts = []
        for bandwidth in bandwidths:
            if isinstance(bandwidth, bool):
                raise ValueError("dropout bandwidths must be positive numbers")
            bandwidth = float(bandwidth)
            if not math.isfinite(bandwidth) or bandwidth <= 0:
                raise ValueError("dropout bandwidths must be positive and finite")

            exact_count = bandwidth / bandwidth_per_quantizer
            count = round(exact_count)
            if not math.isclose(exact_count, count, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"dropout bandwidth {bandwidth:g} kbps is not exactly "
                    f"representable at {frame_rate} frames/s"
                )
            if not 1 <= count <= self.n_q:
                raise ValueError(
                    f"dropout bandwidth {bandwidth:g} kbps requires {count} "
                    f"quantizers, outside the configured range 1..{self.n_q}"
                )
            counts.append(count)

        if len(set(counts)) != len(counts):
            raise ValueError("dropout bandwidths must map to distinct RVQ prefixes")
        return tuple(counts)

    def _active_counts(
        self,
        batch_size: int,
        device: torch.device,
        explicit_count: Optional[int],
        dropout_num_quantizers: Optional[Sequence[int]],
    ) -> torch.Tensor:
        if explicit_count is not None:
            return torch.full(
                (batch_size,), explicit_count, device=device, dtype=torch.long
            )
        counts = torch.full((batch_size,), self.n_q, device=device, dtype=torch.long)
        if self.training and self.quantizer_dropout > 0:
            dropout_count = int(batch_size * self.quantizer_dropout)
            if dropout_count:
                if dropout_num_quantizers is None:
                    candidates = torch.arange(
                        1, self.n_q + 1, device=device, dtype=torch.long
                    )
                else:
                    candidate_values = tuple(dropout_num_quantizers)
                    if not candidate_values:
                        raise ValueError(
                            "dropout_num_quantizers must be a non-empty sequence"
                        )
                    if any(
                        isinstance(value, bool)
                        or int(value) != value
                        or not 1 <= int(value) <= self.n_q
                        for value in candidate_values
                    ):
                        raise ValueError(
                            "dropout_num_quantizers must contain integers between "
                            f"1 and {self.n_q}"
                        )
                    if len(set(candidate_values)) != len(candidate_values):
                        raise ValueError(
                            "dropout_num_quantizers must contain distinct prefixes"
                        )
                    candidates = torch.tensor(
                        candidate_values, device=device, dtype=torch.long
                    )
                indices = torch.randperm(batch_size, device=device)[:dropout_count]
                choices = torch.randint(
                    candidates.numel(), (dropout_count,), device=device
                )
                counts[indices] = candidates[choices]
        return counts

    def forward(
        self,
        x: torch.Tensor,
        frame_rate: int,
        bandwidth: Optional[float] = None,
        num_quantizers: Optional[int] = None,
        dropout_num_quantizers: Optional[Sequence[int]] = None,
    ) -> DACQuantizedResult:
        """Quantize input, optionally forcing one deterministic RVQ prefix.

        ``dropout_num_quantizers`` limits random training-time dropout to the
        supplied prefix lengths. It has no effect during evaluation or when an
        explicit ``bandwidth`` or ``num_quantizers`` is supplied.
        """
        if bandwidth is not None and num_quantizers is not None:
            raise ValueError("Specify bandwidth or num_quantizers, not both")
        explicit_count = num_quantizers
        if bandwidth is not None:
            explicit_count = self.get_num_quantizers_for_bandwidth(
                frame_rate, bandwidth
            )
        if explicit_count is not None and not 1 <= explicit_count <= self.n_q:
            raise ValueError("num_quantizers must be between 1 and the RVQ depth")

        counts = self._active_counts(
            x.shape[0],
            x.device,
            explicit_count,
            dropout_num_quantizers,
        )
        quantized, codes, commitment_loss, codebook_loss = self.vq(x, counts)
        bandwidths = counts.to(x.dtype) * self.get_bandwidth_per_quantizer(frame_rate)
        return DACQuantizedResult(
            quantized=quantized,
            codes=codes,
            bandwidth=bandwidths,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            num_quantizers=counts,
        )

    def encode(
        self,
        x: torch.Tensor,
        frame_rate: int,
        bandwidth: Optional[float] = None,
        num_quantizers: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode deterministically; all layers are used when no target is given."""
        if bandwidth is not None and num_quantizers is not None:
            raise ValueError("Specify bandwidth or num_quantizers, not both")
        count = num_quantizers
        if count is None:
            count = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        return self.vq.encode(x, count)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.vq.decode(codes)
