"""Integration tests for the VoCodec generator."""

import pytest
import torch

from espnet2.gan_codec.shared.streamability import assert_codec_streamable
from espnet2.gan_codec.vocodec.generator import (
    STFTAnalysis,
    STFTSynthesis,
    VoCodecGenerator,
)


def tiny_generator(**overrides):
    params = dict(
        sample_rate=8000,
        hidden_dim=8,
        n_fft=32,
        hop_length=8,
        encoder_num_convnext_blocks=1,
        encoder_convnext_hidden_dim=12,
        decoder_num_convnext_blocks=1,
        decoder_convnext_hidden_dim=8,
        posnet_num_blocks=2,
        posnet_dropout=0.0,
        attention_after=1,
        attention_heads=2,
        attention_context_frames=5,
        attention_query_tile_size=4,
        quantizer_codebook_dim=2,
        quantizer_n_q=2,
        quantizer_bins=8,
        quantizer_dropout=0.0,
        quantizer_projection_norm="none",
        quantizer_target_bandwidth=[3.0, 6.0],
    )
    params.update(overrides)
    return VoCodecGenerator(**params)


def test_stft_analysis_synthesis_is_length_preserving():
    torch.manual_seed(0)
    x = torch.randn(2, 1, 2413)
    analysis = STFTAnalysis(n_fft=720, hop_length=240)
    synthesis = STFTSynthesis(n_fft=720, hop_length=240)
    reconstructed = synthesis(analysis(x), length=x.shape[-1])
    assert reconstructed.shape == x.shape
    torch.testing.assert_close(reconstructed, x, atol=1e-5, rtol=1e-5)


def test_generator_forward_encode_and_decode():
    generator = tiny_generator().eval()
    assert generator.dropout_num_quantizers == (1, 2)
    x = torch.randn(2, 1, 128)
    audio, commitment, codebook, dual = generator(x)
    assert audio.shape == x.shape
    assert commitment.ndim == codebook.ndim == 0
    assert dual is None

    codes = generator.encode(x, target_bw=3.0)
    assert codes.shape == (1, 2, 16)
    assert generator.decode(codes).shape == x.shape


def test_dropout_sampling_strategy_is_configurable():
    target_rates = tiny_generator()
    assert target_rates.quantizer_dropout_sampling == "target_bandwidths"
    assert target_rates.dropout_num_quantizers == (1, 2)

    all_layers = tiny_generator(quantizer_dropout_sampling="all_layers")
    assert all_layers.quantizer_dropout_sampling == "all_layers"
    assert all_layers.dropout_num_quantizers is None


def test_invalid_dropout_sampling_strategy_is_rejected():
    with pytest.raises(ValueError, match="quantizer_dropout_sampling"):
        tiny_generator(quantizer_dropout_sampling="invalid")


def test_complete_generator_respects_declared_latency():
    generator = tiny_generator().eval()
    assert_codec_streamable(
        generator,
        torch.randn(1, 1, 256),
        sample_rate=8000,
        max_future_dependency_ms=4.0,
        num_trials=4,
        atol=1e-5,
        rtol=1e-4,
    )
