# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""LRAC 2 convolutional baseline with DAC-style residual quantization."""

import math
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from espnet2.gan_codec.shared.decoder.generic_seanet import GenericSEANetDecoder
from espnet2.gan_codec.shared.encoder.generic_seanet import GenericSEANetEncoder
from espnet2.gan_codec.shared.quantizer.dac_residual_vq import (
    DACResidualVectorQuantizer,
)
from espnet2.gan_codec.vocodec.vocodec import VoCodec


class LRAC2ConvGenerator(nn.Module):
    """LRAC convolutional generator using a trainable DAC-style RVQ."""

    def __init__(
        self,
        sample_rate: int = 24000,
        encoder_params: Optional[Dict[str, Any]] = None,
        decoder_params: Optional[Dict[str, Any]] = None,
        quantizer_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        encoder_params = dict(encoder_params or {})
        decoder_params = dict(decoder_params or {})
        quantizer_params = dict(quantizer_params or {})
        if not encoder_params or not decoder_params:
            raise ValueError("encoder_params and decoder_params must be provided")

        supported_quantizer_params = {
            "quantizer_codebook_dim",
            "quantizer_n_q",
            "quantizer_bins",
            "quantizer_dropout",
            "quantizer_dropout_sampling",
            "quantizer_projection_mode",
            "quantizer_projection_bias",
            "quantizer_projection_norm",
            "quantizer_normalized_lookup",
            "quantizer_loss_normalization",
            "quantizer_target_bandwidth",
        }
        unknown = set(quantizer_params) - supported_quantizer_params
        if unknown:
            raise ValueError(f"Unsupported quantizer parameters: {sorted(unknown)}")

        latent_dim = encoder_params["output_dimension"]
        if decoder_params["input_dimension"] != latent_dim:
            raise ValueError("encoder output and decoder input dimensions must match")

        self.encoder = GenericSEANetEncoder(**encoder_params)
        self.decoder = GenericSEANetDecoder(**decoder_params)
        self.quantizer = DACResidualVectorQuantizer(
            dimension=latent_dim,
            codebook_dim=quantizer_params.get("quantizer_codebook_dim", 5),
            n_q=quantizer_params.get("quantizer_n_q", 32),
            bins=quantizer_params.get("quantizer_bins", 1024),
            quantizer_dropout=quantizer_params.get("quantizer_dropout", 0.5),
            projection_mode=quantizer_params.get(
                "quantizer_projection_mode", "per_layer"
            ),
            projection_bias=quantizer_params.get("quantizer_projection_bias", True),
            projection_norm=quantizer_params.get(
                "quantizer_projection_norm", "weight_norm"
            ),
            normalized_lookup=quantizer_params.get("quantizer_normalized_lookup", True),
            loss_normalization=quantizer_params.get(
                "quantizer_loss_normalization", "full_batch"
            ),
        )
        self.target_bandwidths: List[float] = list(
            quantizer_params.get("quantizer_target_bandwidth", [1, 6, 12, 24, 32])
        )
        if not self.target_bandwidths:
            raise ValueError("quantizer_target_bandwidth must not be empty")
        self.sample_rate = sample_rate
        self.frame_shift = math.prod(encoder_params["strides"])
        if sample_rate % self.frame_shift:
            raise ValueError("sample rate must be divisible by the encoder frame shift")
        self.frame_rate = sample_rate // self.frame_shift

        target_num_quantizers = self.quantizer.get_num_quantizers_for_bandwidths(
            self.frame_rate, self.target_bandwidths
        )
        dropout_sampling = quantizer_params.get(
            "quantizer_dropout_sampling", "target_bandwidths"
        )
        if dropout_sampling == "target_bandwidths":
            self.dropout_num_quantizers = target_num_quantizers
        elif dropout_sampling == "all_layers":
            self.dropout_num_quantizers = None
        else:
            raise ValueError(
                "quantizer_dropout_sampling must be 'target_bandwidths' "
                "or 'all_layers'"
            )
        self.quantizer_dropout_sampling = dropout_sampling

    def forward(self, x: torch.Tensor, use_dual_decoder: bool = False):
        if use_dual_decoder:
            raise ValueError("LRAC2ConvGenerator does not use a dual decoder")
        latent = self.encoder(x)
        result = self.quantizer(
            latent,
            self.frame_rate,
            dropout_num_quantizers=self.dropout_num_quantizers,
        )
        audio = self.decoder(result.quantized)
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


class LRAC2ConvBaseline(VoCodec):
    """LRAC 2 convolutional model with VoCodec discriminator and losses."""

    @staticmethod
    def _build_generator(
        sampling_rate: int, generator_params: Dict[str, Any]
    ) -> torch.nn.Module:
        return LRAC2ConvGenerator(sample_rate=sampling_rate, **generator_params)
