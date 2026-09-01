"""Tests for configurable multi-scale mel-loss aggregation."""

import pytest
import torch
from torch import nn

from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss


class FixedMelLoss(nn.Module):
    def forward(self, y_hat, y, use_mse=False):
        return y_hat.new_tensor(2.0 if use_mse else 1.0)


def configured_loss(use_mse, scale_reduction):
    loss = MultiScaleMelSpectrogramLoss.__new__(MultiScaleMelSpectrogramLoss)
    nn.Module.__init__(loss)
    loss.mel_loss = nn.ModuleList([FixedMelLoss(), FixedMelLoss()])
    loss.alphas = [1.0, 2.0]
    loss.total = 5.0
    loss.use_mse = use_mse
    loss.scale_reduction = scale_reduction
    return loss


def test_legacy_normalized_l1_plus_mse_behavior():
    loss = configured_loss(use_mse=True, scale_reduction="normalized")
    value = loss(torch.zeros(1), torch.zeros(1))
    torch.testing.assert_close(value, torch.tensor(1.6))


def test_vocodec_sums_l1_scales_without_mse():
    loss = configured_loss(use_mse=False, scale_reduction="sum")
    value = loss(torch.zeros(1), torch.zeros(1))
    torch.testing.assert_close(value, torch.tensor(2.0))


def test_scalar_n_mels_is_broadcast_to_all_scales():
    loss = MultiScaleMelSpectrogramLoss(
        fs=120,
        range_start=3,
        range_end=5,
        n_mels=2,
    )
    assert len(loss.mel_loss) == 2


def test_n_mels_must_cover_every_scale():
    with pytest.raises(ValueError, match="at least 2 values"):
        MultiScaleMelSpectrogramLoss(
            fs=120,
            range_start=3,
            range_end=5,
            n_mels=[2],
        )


def test_invalid_scale_reduction_is_rejected():
    with pytest.raises(ValueError, match="scale_reduction"):
        MultiScaleMelSpectrogramLoss(scale_reduction="invalid")
