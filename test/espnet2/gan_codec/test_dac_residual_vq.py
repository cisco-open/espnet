# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Tests for the trainable DAC-style RVQ."""

import pytest
import torch

from espnet2.gan_codec.shared.quantizer.dac_residual_vq import (
    DACResidualVectorQuantizer,
)


def make_quantizer(**kwargs):
    defaults = dict(
        dimension=4,
        codebook_dim=2,
        n_q=4,
        bins=8,
        quantizer_dropout=0.5,
        projection_norm="none",
    )
    defaults.update(kwargs)
    return DACResidualVectorQuantizer(**defaults)


def test_explicit_quantizer_count_is_respected_during_training():
    quantizer = make_quantizer().train()
    result = quantizer(
        torch.randn(3, 4, 6),
        frame_rate=100,
        num_quantizers=2,
        dropout_num_quantizers=(1, 4),
    )
    assert result.codes.shape == (4, 3, 6)
    assert result.num_quantizers.tolist() == [2, 2, 2]
    assert torch.all(result.codes[2:] == -1)


def test_evaluation_is_deterministic_and_encode_uses_requested_prefix():
    quantizer = make_quantizer(quantizer_dropout=0.9).eval()
    x = torch.randn(2, 4, 6)
    first = quantizer(x, frame_rate=100, dropout_num_quantizers=(1, 3))
    second = quantizer(x, frame_rate=100, dropout_num_quantizers=(1, 3))
    torch.testing.assert_close(first.quantized, second.quantized)
    assert first.num_quantizers.tolist() == [4, 4]
    assert quantizer.encode(x, frame_rate=100, bandwidth=0.31).shape[0] == 1
    assert quantizer.encode(x, frame_rate=100, bandwidth=0.91).shape[0] == 3


def test_dropout_probability_semantics():
    torch.manual_seed(0)
    quantizer = make_quantizer(n_q=8, quantizer_dropout=0.5).train()
    result = quantizer(torch.randn(2000, 4, 1), frame_rate=100)
    random_prefix_fraction = (result.num_quantizers < 8).float().mean()
    # Some selected dropout examples independently draw the full prefix.
    assert 0.35 < random_prefix_fraction < 0.55


def test_dropout_without_candidates_samples_all_layer_counts():
    torch.manual_seed(0)
    quantizer = make_quantizer(n_q=8, quantizer_dropout=1.0).train()
    counts = quantizer._active_counts(
        batch_size=2000,
        device=torch.device("cpu"),
        explicit_count=None,
        dropout_num_quantizers=None,
    )
    assert set(counts.tolist()) == set(range(1, 9))


def test_dropout_samples_only_configured_prefixes_uniformly():
    torch.manual_seed(0)
    quantizer = make_quantizer(n_q=32, quantizer_dropout=1.0).train()
    candidates = (1, 6, 12, 24, 32)
    counts = quantizer._active_counts(
        batch_size=5000,
        device=torch.device("cpu"),
        explicit_count=None,
        dropout_num_quantizers=candidates,
    )

    assert set(counts.tolist()) == set(candidates)
    for candidate in candidates:
        fraction = (counts == candidate).float().mean()
        assert 0.17 < fraction < 0.23


def test_half_dropout_gives_expected_overall_target_distribution():
    torch.manual_seed(0)
    quantizer = make_quantizer(n_q=32, quantizer_dropout=0.5).train()
    candidates = (1, 6, 12, 24, 32)
    counts = quantizer._active_counts(
        batch_size=10000,
        device=torch.device("cpu"),
        explicit_count=None,
        dropout_num_quantizers=candidates,
    )

    expected = {1: 0.1, 6: 0.1, 12: 0.1, 24: 0.1, 32: 0.6}
    for candidate, expected_fraction in expected.items():
        fraction = (counts == candidate).float().mean()
        assert abs(fraction.item() - expected_fraction) < 0.02


def test_explicit_bandwidth_disables_target_rate_sampling():
    quantizer = make_quantizer(n_q=4, quantizer_dropout=1.0).train()
    result = quantizer(
        torch.randn(8, 4, 2),
        frame_rate=100,
        bandwidth=0.6,
        dropout_num_quantizers=(1, 4),
    )
    assert result.num_quantizers.tolist() == [2] * 8


def test_target_bandwidths_map_to_100hz_and_50hz_prefixes():
    quantizer = make_quantizer(n_q=64, bins=1024)
    bandwidths = [1, 6, 12, 24, 32]
    assert quantizer.get_num_quantizers_for_bandwidths(100, bandwidths) == (
        1,
        6,
        12,
        24,
        32,
    )
    assert quantizer.get_num_quantizers_for_bandwidths(50, bandwidths) == (
        2,
        12,
        24,
        48,
        64,
    )


@pytest.mark.parametrize(
    "bandwidths, message",
    [
        ([], "non-empty"),
        ([1.5], "not exactly representable"),
        ([33], "outside the configured range"),
        ([1, 1], "distinct RVQ prefixes"),
    ],
)
def test_invalid_dropout_bandwidths_are_rejected(bandwidths, message):
    quantizer = make_quantizer(n_q=32, bins=1024)
    with pytest.raises(ValueError, match=message):
        quantizer.get_num_quantizers_for_bandwidths(100, bandwidths)


def test_invalid_dropout_prefixes_are_rejected():
    quantizer = make_quantizer(n_q=4, quantizer_dropout=1.0).train()
    with pytest.raises(ValueError, match="integers between 1 and 4"):
        quantizer(torch.randn(2, 4, 1), 100, dropout_num_quantizers=(1, 5))
    with pytest.raises(ValueError, match="distinct prefixes"):
        quantizer(torch.randn(2, 4, 1), 100, dropout_num_quantizers=(1, 1))


@pytest.mark.parametrize("mode", ["per_layer", "shared"])
def test_projection_modes_round_trip_shapes(mode):
    quantizer = make_quantizer(projection_mode=mode).eval()
    x = torch.randn(2, 4, 6)
    codes = quantizer.encode(x, frame_rate=100, num_quantizers=3)
    assert codes.shape == (3, 2, 6)
    assert quantizer.decode(codes).shape == x.shape


def test_none_projection_requires_matching_dimensions():
    with pytest.raises(ValueError):
        make_quantizer(projection_mode="none")
    quantizer = make_quantizer(
        dimension=2, codebook_dim=2, projection_mode="none"
    ).eval()
    assert quantizer(torch.randn(1, 2, 3), 100).quantized.shape == (1, 2, 3)


def test_stop_gradient_directions():
    quantizer = make_quantizer(
        dimension=2,
        codebook_dim=2,
        n_q=1,
        projection_mode="none",
        quantizer_dropout=0.0,
    ).train()

    x = torch.randn(2, 2, 5, requires_grad=True)
    result = quantizer(x, 100)
    result.quantized.square().mean().backward()
    assert x.grad is not None
    assert quantizer.vq.layers[0].codebook.weight.grad is None

    quantizer.zero_grad(set_to_none=True)
    x = torch.randn(2, 2, 5, requires_grad=True)
    result = quantizer(x, 100)
    result.codebook_loss.backward()
    assert x.grad is None
    assert quantizer.vq.layers[0].codebook.weight.grad is not None

    quantizer.zero_grad(set_to_none=True)
    x = torch.randn(2, 2, 5, requires_grad=True)
    result = quantizer(x, 100)
    result.commitment_loss.backward()
    assert x.grad is not None
    assert quantizer.vq.layers[0].codebook.weight.grad is None


def test_fully_inactive_layers_receive_zero_not_missing_gradients():
    quantizer = make_quantizer(quantizer_dropout=0.0).train()
    x = torch.randn(2, 4, 5)
    counts = torch.ones(2, dtype=torch.long)
    quantized, _, commitment, codebook = quantizer.vq(x, counts)
    (quantized.mean() + commitment + codebook).backward()
    for layer in quantizer.vq.layers[1:]:
        for parameter in layer.parameters():
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("value", [-0.1, 1.1, True])
def test_dropout_requires_explicit_probability(value):
    with pytest.raises(ValueError):
        make_quantizer(quantizer_dropout=value)
