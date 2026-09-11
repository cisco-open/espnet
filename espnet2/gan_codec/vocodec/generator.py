"""VoCodec time-frequency generator."""

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from espnet2.gan_codec.shared.encoder.seanet import ConvLayerNorm, SConv1d
from espnet2.gan_codec.shared.quantizer.dac_residual_vq import (
    DACResidualVectorQuantizer,
)
from espnet2.gan_codec.vocodec.blocks import ConvNeXtBlock1d, PosNet


PosNetGroups = Union[Tuple[int, int], Sequence[Tuple[int, int]]]


class STFTAnalysis(nn.Module):
    """Square-root-Hann STFT with 30 ms window and 10 ms hop by default."""

    def __init__(self, n_fft: int = 720, hop_length: int = 240):
        super().__init__()
        if n_fft < hop_length or (n_fft - hop_length) % 2:
            raise ValueError("n_fft - hop_length must be non-negative and even")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.edge_padding = (n_fft - hop_length) // 2
        self.register_buffer("window", torch.hann_window(n_fft).sqrt())

    @property
    def feature_dim(self) -> int:
        return 2 * (self.n_fft // 2 + 1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError("audio must have shape (batch, 1, samples)")
        remainder = audio.shape[-1] % self.hop_length
        end_padding = (self.hop_length - remainder) % self.hop_length
        audio = F.pad(
            audio[:, 0],
            (self.edge_padding, self.edge_padding + end_padding),
        )
        spectrum = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )
        log_magnitude = spectrum.abs().clamp_min(1e-7).log()
        return torch.cat((log_magnitude, spectrum.angle()), dim=1)


class STFTSynthesis(nn.Module):
    """Overlap-add inverse matching :class:`STFTAnalysis`."""

    def __init__(
        self,
        n_fft: int = 720,
        hop_length: int = 240,
        max_magnitude: float = 100.0,
    ):
        super().__init__()
        if n_fft < hop_length or (n_fft - hop_length) % 2:
            raise ValueError("n_fft - hop_length must be non-negative and even")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.edge_padding = (n_fft - hop_length) // 2
        self.max_log_magnitude = math.log(max_magnitude)
        self.register_buffer("window", torch.hann_window(n_fft).sqrt())

    def _overlap_add(self, frames: torch.Tensor) -> torch.Tensor:
        batch, _, num_frames = frames.shape
        full_length = (num_frames - 1) * self.hop_length + self.n_fft
        audio = F.fold(
            frames,
            output_size=(1, full_length),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).squeeze(2)
        envelope_frames = self.window.square()[None, :, None].expand(
            batch, -1, num_frames
        )
        envelope = F.fold(
            envelope_frames,
            output_size=(1, full_length),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).squeeze(2)
        return audio / envelope.clamp_min(1e-8)

    def forward(
        self, features: torch.Tensor, length: Optional[int] = None
    ) -> torch.Tensor:
        bins = self.n_fft // 2 + 1
        if features.ndim != 3 or features.shape[1] != 2 * bins:
            raise ValueError("features have an incompatible STFT dimension")
        log_magnitude, phase = features.split(bins, dim=1)
        magnitude = log_magnitude.clamp(max=self.max_log_magnitude).exp()
        spectrum = torch.polar(magnitude, phase)
        frames = (
            torch.fft.irfft(spectrum, n=self.n_fft, dim=1) * self.window[None, :, None]
        )
        audio = self._overlap_add(frames)
        start = self.edge_padding
        output_length = features.shape[-1] * self.hop_length
        audio = audio[..., start : start + output_length]
        return audio[..., :length] if length is not None else audio


class VoCodecEncoder(nn.Module):
    """VoCodec STFT encoder."""

    def __init__(
        self,
        hidden_dim: int = 192,
        n_fft: int = 720,
        hop_length: int = 240,
        num_convnext_blocks: int = 6,
        convnext_hidden_dim: int = 216,
        convnext_kernel_size: int = 7,
        convnext_layer_scale_init_value: Optional[float] = None,
        posnet_num_blocks: int = 4,
        posnet_kernel_size: int = 3,
        posnet_groups: PosNetGroups = (1, 1),
        posnet_dropout: float = 0.1,
        attention_after: int = 2,
        attention_heads: int = 4,
        attention_context_frames: int = 300,
        attention_lookahead_frames: int = 0,
        attention_query_tile_size: int = 64,
        attention_use_rope: bool = False,
        attention_rotary_dim: Optional[int] = None,
        attention_rope_base: float = 10000.0,
        latent_kernel_size: int = 3,
    ):
        super().__init__()
        self.analysis = STFTAnalysis(n_fft, hop_length)
        self.input_projection = nn.Conv1d(
            self.analysis.feature_dim, hidden_dim, kernel_size=1
        )
        self.input_norm = ConvLayerNorm(hidden_dim)
        self.convnext = nn.Sequential(
            *[
                ConvNeXtBlock1d(
                    hidden_dim,
                    convnext_hidden_dim,
                    kernel_size=convnext_kernel_size,
                    layer_scale_init_value=convnext_layer_scale_init_value,
                )
                for _ in range(num_convnext_blocks)
            ]
        )
        self.posnet = PosNet(
            hidden_dim,
            num_blocks=posnet_num_blocks,
            kernel_size=posnet_kernel_size,
            groups=tuple(posnet_groups),
            dropout=posnet_dropout,
            attention_after=attention_after,
            attention_heads=attention_heads,
            attention_context_frames=attention_context_frames,
            attention_lookahead_frames=attention_lookahead_frames,
            attention_query_tile_size=attention_query_tile_size,
            attention_use_rope=attention_use_rope,
            attention_rotary_dim=attention_rotary_dim,
            attention_rope_base=attention_rope_base,
        )
        self.latent_projection = SConv1d(
            hidden_dim,
            hidden_dim,
            latent_kernel_size,
            causal=True,
            norm="none",
            pad_mode="constant",
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(self.input_projection(self.analysis(audio)))
        return self.latent_projection(self.posnet(self.convnext(x)))


class VoCodecDecoder(nn.Module):
    """VoCodec STFT decoder with non-inverted ConvNeXt blocks."""

    def __init__(
        self,
        hidden_dim: int = 192,
        n_fft: int = 720,
        hop_length: int = 240,
        num_convnext_blocks: int = 6,
        convnext_hidden_dim: int = 192,
        convnext_kernel_size: int = 7,
        convnext_layer_scale_init_value: Optional[float] = None,
        posnet_num_blocks: int = 4,
        posnet_kernel_size: int = 3,
        posnet_groups: PosNetGroups = (2, 2),
        posnet_dropout: float = 0.1,
        attention_after: int = 2,
        attention_heads: int = 4,
        attention_context_frames: int = 300,
        attention_lookahead_frames: int = 0,
        attention_query_tile_size: int = 64,
        attention_use_rope: bool = False,
        attention_rotary_dim: Optional[int] = None,
        attention_rope_base: float = 10000.0,
        latent_kernel_size: int = 3,
        max_magnitude: float = 100.0,
    ):
        super().__init__()
        self.latent_projection = SConv1d(
            hidden_dim,
            hidden_dim,
            latent_kernel_size,
            causal=True,
            norm="none",
            pad_mode="constant",
        )
        self.posnet = PosNet(
            hidden_dim,
            num_blocks=posnet_num_blocks,
            kernel_size=posnet_kernel_size,
            groups=tuple(posnet_groups),
            dropout=posnet_dropout,
            attention_after=attention_after,
            attention_heads=attention_heads,
            attention_context_frames=attention_context_frames,
            attention_lookahead_frames=attention_lookahead_frames,
            attention_query_tile_size=attention_query_tile_size,
            attention_use_rope=attention_use_rope,
            attention_rotary_dim=attention_rotary_dim,
            attention_rope_base=attention_rope_base,
        )
        self.convnext = nn.Sequential(
            *[
                ConvNeXtBlock1d(
                    hidden_dim,
                    convnext_hidden_dim,
                    kernel_size=convnext_kernel_size,
                    layer_scale_init_value=convnext_layer_scale_init_value,
                )
                for _ in range(num_convnext_blocks)
            ]
        )
        self.output_norm = ConvLayerNorm(hidden_dim)
        self.synthesis = STFTSynthesis(n_fft, hop_length, max_magnitude)
        self.output_projection = nn.Conv1d(
            hidden_dim, 2 * (n_fft // 2 + 1), kernel_size=1
        )

    def forward(
        self, latent: torch.Tensor, length: Optional[int] = None
    ) -> torch.Tensor:
        x = self.latent_projection(latent)
        x = self.output_projection(self.output_norm(self.convnext(self.posnet(x))))
        return self.synthesis(x, length=length)


class VoCodecGenerator(nn.Module):
    """VoCodec generator with DAC-style RVQ."""

    def __init__(
        self,
        sample_rate: int = 24000,
        hidden_dim: int = 192,
        n_fft: int = 720,
        hop_length: int = 240,
        encoder_num_convnext_blocks: int = 6,
        encoder_convnext_hidden_dim: int = 216,
        decoder_num_convnext_blocks: int = 6,
        decoder_convnext_hidden_dim: int = 192,
        convnext_kernel_size: int = 7,
        convnext_layer_scale_init_value: Optional[float] = None,
        posnet_num_blocks: int = 4,
        posnet_kernel_size: int = 3,
        encoder_posnet_groups: PosNetGroups = (1, 1),
        decoder_posnet_groups: PosNetGroups = (2, 2),
        posnet_dropout: float = 0.1,
        attention_after: int = 2,
        attention_heads: int = 4,
        attention_context_frames: int = 300,
        attention_lookahead_frames: int = 0,
        attention_query_tile_size: int = 64,
        attention_use_rope: bool = False,
        attention_rotary_dim: Optional[int] = None,
        attention_rope_base: float = 10000.0,
        latent_kernel_size: int = 3,
        quantizer_codebook_dim: int = 8,
        quantizer_n_q: int = 32,
        quantizer_bins: int = 1024,
        quantizer_dropout: float = 0.5,
        quantizer_dropout_sampling: str = "target_bandwidths",
        quantizer_projection_mode: str = "per_layer",
        quantizer_projection_bias: bool = True,
        quantizer_projection_norm: str = "weight_norm",
        quantizer_normalized_lookup: bool = True,
        quantizer_loss_normalization: str = "full_batch",
        quantizer_target_bandwidth: List[float] = [1, 6, 12, 24, 32],
        max_magnitude: float = 100.0,
    ):
        super().__init__()
        if sample_rate % hop_length:
            raise ValueError("sample_rate must be divisible by hop_length")
        common = dict(
            hidden_dim=hidden_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            convnext_kernel_size=convnext_kernel_size,
            convnext_layer_scale_init_value=convnext_layer_scale_init_value,
            posnet_num_blocks=posnet_num_blocks,
            posnet_kernel_size=posnet_kernel_size,
            posnet_dropout=posnet_dropout,
            attention_after=attention_after,
            attention_heads=attention_heads,
            attention_context_frames=attention_context_frames,
            attention_lookahead_frames=attention_lookahead_frames,
            attention_query_tile_size=attention_query_tile_size,
            attention_use_rope=attention_use_rope,
            attention_rotary_dim=attention_rotary_dim,
            attention_rope_base=attention_rope_base,
            latent_kernel_size=latent_kernel_size,
        )
        self.encoder = VoCodecEncoder(
            num_convnext_blocks=encoder_num_convnext_blocks,
            convnext_hidden_dim=encoder_convnext_hidden_dim,
            posnet_groups=encoder_posnet_groups,
            **common,
        )
        self.decoder = VoCodecDecoder(
            num_convnext_blocks=decoder_num_convnext_blocks,
            convnext_hidden_dim=decoder_convnext_hidden_dim,
            posnet_groups=decoder_posnet_groups,
            max_magnitude=max_magnitude,
            **common,
        )
        self.quantizer = DACResidualVectorQuantizer(
            dimension=hidden_dim,
            codebook_dim=quantizer_codebook_dim,
            n_q=quantizer_n_q,
            bins=quantizer_bins,
            quantizer_dropout=quantizer_dropout,
            projection_mode=quantizer_projection_mode,
            projection_bias=quantizer_projection_bias,
            projection_norm=quantizer_projection_norm,
            normalized_lookup=quantizer_normalized_lookup,
            loss_normalization=quantizer_loss_normalization,
        )
        self.sample_rate = sample_rate
        self.frame_rate = sample_rate // hop_length
        self.frame_shift = hop_length
        self.target_bandwidths = list(quantizer_target_bandwidth)
        target_num_quantizers = self.quantizer.get_num_quantizers_for_bandwidths(
            self.frame_rate, self.target_bandwidths
        )
        if quantizer_dropout_sampling == "target_bandwidths":
            self.dropout_num_quantizers = target_num_quantizers
        elif quantizer_dropout_sampling == "all_layers":
            self.dropout_num_quantizers = None
        else:
            raise ValueError(
                "quantizer_dropout_sampling must be 'target_bandwidths' "
                "or 'all_layers'"
            )
        self.quantizer_dropout_sampling = quantizer_dropout_sampling

    def forward(self, x: torch.Tensor, use_dual_decoder: bool = False):
        if use_dual_decoder:
            raise ValueError("VoCodec does not use a dual decoder")
        latent = self.encoder(x)
        result = self.quantizer(
            latent,
            self.frame_rate,
            dropout_num_quantizers=self.dropout_num_quantizers,
        )
        audio = self.decoder(result.quantized, length=x.shape[-1])
        return audio, result.commitment_loss, result.codebook_loss, None

    def encode(
        self, x: torch.Tensor, target_bw: Optional[float] = None
    ) -> torch.Tensor:
        if target_bw is None:
            target_bw = self.target_bandwidths[-1]
        latent = self.encoder(x)
        return self.quantizer.encode(
            latent, self.frame_rate, bandwidth=float(target_bw)
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.quantizer.decode(codes))
