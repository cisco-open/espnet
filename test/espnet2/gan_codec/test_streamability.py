# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Tests for the public sample-based streamability check."""

import pytest
import torch
from torch import nn

from espnet2.gan_codec.shared.streamability import (
    assert_codec_streamable,
    check_codec_streamability,
)


class Lookahead(nn.Module):
    def __init__(self, samples: int):
        super().__init__()
        self.samples = samples

    def forward(self, x):
        return torch.nn.functional.pad(x[..., self.samples :], (0, self.samples))


class UnboundedFuture(nn.Module):
    def forward(self, x):
        return x + x[..., -1:]


class CropStartup(nn.Module):
    def forward(self, x):
        return x[..., 3:]


def test_declared_lookahead_passes_and_restores_training_mode():
    model = Lookahead(samples=4).train()
    result = assert_codec_streamable(
        model,
        torch.randn(2, 1, 64),
        sample_rate=1000,
        max_future_dependency_ms=4,
        num_trials=5,
    )
    assert result.passed
    assert result.allowed_future_samples == 4
    assert result.absolute_tolerance == 1e-6
    assert result.relative_tolerance == 1e-5
    assert model.training


def test_unbounded_future_fails():
    result = check_codec_streamability(
        UnboundedFuture(),
        torch.randn(1, 1, 64),
        sample_rate=1000,
        max_future_dependency_ms=4,
        num_trials=5,
    )
    assert not result.passed


def test_explicit_startup_crop_uses_sample_alignment():
    result = check_codec_streamability(
        CropStartup(),
        torch.randn(1, 1, 64),
        sample_rate=1000,
        max_future_dependency_ms=0,
        crop_left_samples=3,
        num_trials=5,
    )
    assert result.passed


def test_30ms_allowance_rejects_31ms_lookahead():
    waveform = torch.randn(1, 1, 256)
    assert check_codec_streamability(
        Lookahead(samples=30),
        waveform,
        sample_rate=1000,
        max_future_dependency_ms=30,
        num_trials=16,
        atol=0.0,
        rtol=0.0,
    ).passed
    assert not check_codec_streamability(
        Lookahead(samples=31),
        waveform,
        sample_rate=1000,
        max_future_dependency_ms=30,
        num_trials=16,
        atol=0.0,
        rtol=0.0,
    ).passed


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("atol", -1.0, "non-negative"),
        ("rtol", -1.0, "non-negative"),
        ("atol", float("inf"), "finite"),
        ("rtol", float("nan"), "finite"),
    ),
)
def test_invalid_tolerance_is_rejected(name, value, message):
    kwargs = {name: value}
    with pytest.raises(ValueError, match=message):
        check_codec_streamability(
            Lookahead(samples=0),
            torch.randn(1, 1, 64),
            sample_rate=1000,
            **kwargs,
        )
