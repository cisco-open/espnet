# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Tests for the LRAC 2 convolutional DAC-RVQ generator."""

import pytest
import torch

from espnet2.gan_codec.lrac2.lrac2 import LRAC2ConvGenerator
from espnet2.gan_codec.shared.streamability import assert_codec_streamable
from espnet2.tasks.gan_codec import codec_choices


def tiny_generator(**quantizer_overrides):
    quantizer_params = {
        "quantizer_codebook_dim": 2,
        "quantizer_n_q": 4,
        "quantizer_bins": 8,
        "quantizer_dropout": 0.5,
        "quantizer_projection_norm": "none",
        "quantizer_target_bandwidth": [3.0, 6.0, 9.0, 12.0],
    }
    quantizer_params.update(quantizer_overrides)
    return LRAC2ConvGenerator(
        sample_rate=8000,
        encoder_params={
            "output_dimension": 4,
            "enable_output_layer": False,
            "n_filters": [2, 4],
            "strides": [8],
            "downsampling_kernel_sizes": [8],
            "residual_kernel_sizes": [[[3, 1]]],
            "residual_dilations": [[[1, 1]]],
            "residual_causality_modes": [[[True, True]]],
            "downsampling_causality_modes": [True],
            "norm": "none",
        },
        decoder_params={
            "input_dimension": 4,
            "output_dimension": 1,
            "enable_input_layer": False,
            "n_filters": [4, 2],
            "strides": [8],
            "upsampling_kernel_sizes": [8],
            "residual_kernel_sizes": [[[3, 1]]],
            "residual_dilations": [[[1, 1]]],
            "residual_causality_modes": [[[True, True]]],
            "upsampling_causality_modes": [True],
            "norm": "none",
        },
        quantizer_params=quantizer_params,
    )


def test_generator_forward_and_requested_bitrates():
    generator = tiny_generator().eval()
    assert generator.dropout_num_quantizers == (1, 2, 3, 4)
    audio = torch.randn(2, 1, 128)

    reconstructed, commitment, codebook, dual = generator(audio)
    assert reconstructed.shape == audio.shape
    assert commitment.ndim == codebook.ndim == 0
    assert dual is None

    for index, bandwidth in enumerate(generator.target_bandwidths, start=1):
        codes = generator.encode(audio, bandwidth)
        assert codes.shape == (index, 2, 16)
        assert generator.decode(codes).shape == audio.shape


def test_dropout_sampling_strategy_is_configurable():
    target_rates = tiny_generator()
    assert target_rates.quantizer_dropout_sampling == "target_bandwidths"
    assert target_rates.dropout_num_quantizers == (1, 2, 3, 4)

    all_layers = tiny_generator(quantizer_dropout_sampling="all_layers")
    assert all_layers.quantizer_dropout_sampling == "all_layers"
    assert all_layers.dropout_num_quantizers is None


def test_invalid_dropout_sampling_strategy_is_rejected():
    with pytest.raises(ValueError, match="quantizer_dropout_sampling"):
        tiny_generator(quantizer_dropout_sampling="invalid")


def test_unknown_quantizer_parameter_is_rejected():
    with pytest.raises(ValueError, match="Unsupported quantizer parameters"):
        tiny_generator(quantizer_typo=1)


def test_nonrepresentable_target_bandwidth_is_rejected():
    with pytest.raises(ValueError, match="not exactly representable"):
        tiny_generator(quantizer_target_bandwidth=[3.5])


def test_complete_generator_has_bounded_future_dependency():
    assert_codec_streamable(
        tiny_generator().eval(),
        torch.randn(1, 1, 256),
        sample_rate=8000,
        max_future_dependency_ms=1.0,
        num_trials=8,
        atol=0.0,
        rtol=0.0,
    )


def test_lrac2_codecs_are_registered():
    assert codec_choices.get_class("lrac2_conv").__name__ == "LRAC2ConvBaseline"
    assert codec_choices.get_class("vocodec").__name__ == "VoCodec"
