#!/usr/bin/env python3
"""Generate and verify the final LRAC 2 Track 2 VoCodec design workbook."""

import sys
from pathlib import Path


RECIPE_DIR = Path(__file__).resolve().parents[1]
LRAC2_COMMON_DIR = RECIPE_DIR.parent / "common"
sys.path.insert(0, str(LRAC2_COMMON_DIR))

from vocodec_design_sheet import (  # noqa: E402
    Variant,
    create_workbook,
    validate_config_matches_variant,
)


VARIANT = Variant(
    track=2,
    label="100 Hz / 24-layer / RoPE / 1.5M cap",
    hop_size=240,
    attention_context_frames=300,
    attention_lookahead_frames=0,
    attention_heads=4,
    rvq_layers=24,
    model_dim=112,
    encoder_convnext_expansion_dim=128,
    decoder_convnext_expansion_dim=128,
    decoder_posnet_groups_first=(1, 1),
    decoder_posnet_groups_last=(2, 2),
    attention_use_rope=True,
    attention_rotary_dim=28,
    supported_bitrates_kbps=(1, 6, 12, 24),
    parameter_budget=1_500_000,
)
OUTPUT = RECIPE_DIR / "LRAC2-Track2-VoCodec-Model-Design-Sheet.xlsx"
CONFIG = RECIPE_DIR / "conf" / "train.yaml"


def main():
    validate_config_matches_variant(VARIANT, CONFIG)
    results = create_workbook(VARIANT, OUTPUT)
    print(f"Created {OUTPUT}")
    for name, value in results.items():
        print(f"{VARIANT.label} {name}: {value}")


if __name__ == "__main__":
    main()
