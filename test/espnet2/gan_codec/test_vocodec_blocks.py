"""Tests for bounded-context VoCodec blocks."""

import pytest
import torch

from espnet2.gan_codec.shared.encoder.seanet import ConvLayerNorm
from espnet2.gan_codec.vocodec.blocks import (
    BoundedMultiheadAttention,
    CausalResNetBlock1d,
    ConvNeXtBlock1d,
    PosNet,
)


def test_conv_layer_norm_returns_tensor():
    x = torch.randn(2, 8, 12)
    assert ConvLayerNorm(8)(x).shape == x.shape


@pytest.mark.parametrize("use_rope", [False, True])
def test_attention_tiled_matches_dense_reference(use_rope):
    torch.manual_seed(0)
    x = torch.randn(2, 8, 19)
    attention = BoundedMultiheadAttention(
        8,
        num_heads=2,
        context_frames=5,
        lookahead_frames=1,
        query_tile_size=4,
        use_rope=use_rope,
        rotary_dim=4,
    ).eval()
    actual = attention(x)
    expected = attention.forward_reference(x)
    torch.testing.assert_close(actual, expected)


def test_rope_absolute_qk_rotation_matches_relative_rotation():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 7)
    attention = BoundedMultiheadAttention(
        8, num_heads=2, context_frames=5, use_rope=True, rotary_dim=4
    ).eval()
    raw_q, raw_k, raw_v = attention._project_qkv(x)
    q, k, v = attention._qkv(x)

    query_position = 5
    key_position = 2
    absolute_score = (q[..., query_position, :] * k[..., key_position, :]).sum(dim=-1)
    relative_key = attention._apply_rope(
        raw_k[..., key_position : key_position + 1, :],
        torch.tensor([key_position - query_position]),
    )
    relative_score = (raw_q[..., query_position, :] * relative_key[..., 0, :]).sum(
        dim=-1
    )

    torch.testing.assert_close(absolute_score, relative_score)
    torch.testing.assert_close(v, raw_v)


def test_rope_is_invariant_to_a_shared_position_offset():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 11)
    attention = BoundedMultiheadAttention(
        8, num_heads=2, context_frames=5, use_rope=True, rotary_dim=4
    ).eval()
    torch.testing.assert_close(
        attention(x, position_offset=0),
        attention(x, position_offset=37),
        atol=1e-6,
        rtol=1e-5,
    )


@pytest.mark.parametrize("rotary_dim", [0, 3, 6])
def test_rope_rejects_invalid_rotary_dimension(rotary_dim):
    with pytest.raises(ValueError, match="rotary_dim"):
        BoundedMultiheadAttention(8, num_heads=2, use_rope=True, rotary_dim=rotary_dim)


def test_attention_has_bounded_past_and_future():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 20)
    attention = BoundedMultiheadAttention(
        8, num_heads=2, context_frames=5, lookahead_frames=0, query_tile_size=3
    ).eval()
    reference = attention(x)

    changed_future = x.clone()
    changed_future[..., 10:] = torch.randn_like(changed_future[..., 10:])
    torch.testing.assert_close(attention(changed_future)[..., :10], reference[..., :10])

    changed_old_history = x.clone()
    changed_old_history[..., 0] += 100
    torch.testing.assert_close(
        attention(changed_old_history)[..., 5:], reference[..., 5:]
    )


def test_convolutional_blocks_are_causal():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 20)
    blocks = [
        ConvNeXtBlock1d(8, 12, kernel_size=7),
        CausalResNetBlock1d(8, kernel_size=3, groups=(2, 2), dropout=0.0),
    ]
    for block in blocks:
        block.eval()
        reference = block(x)
        changed = x.clone()
        changed[..., 10:] = torch.randn_like(changed[..., 10:])
        torch.testing.assert_close(block(changed)[..., :10], reference[..., :10])


def test_posnet_accepts_per_block_groups():
    posnet = PosNet(
        8,
        num_blocks=4,
        groups=((1, 1), (1, 1), (2, 2), (2, 2)),
        attention_heads=2,
        attention_context_frames=5,
    )
    groups = [block.conv1.conv.conv.groups for block in posnet.blocks]
    assert groups == [1, 1, 2, 2]
