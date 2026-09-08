#!/usr/bin/env python3
"""Generate and verify the final LRAC 2 Track 1 VoCodec design workbook."""

import sys
from pathlib import Path


RECIPE_DIR = Path(__file__).resolve().parents[1]
LRAC2_COMMON_DIR = RECIPE_DIR.parent / "common"
sys.path.insert(0, str(LRAC2_COMMON_DIR))

from vocodec_design_sheet import Variant, create_workbook  # noqa: E402


VARIANT = Variant(
    track=1,
    label="100 Hz / 32-layer",
    hop_size=240,
    attention_context_frames=300,
    attention_lookahead_frames=0,
    attention_heads=4,
    rvq_layers=32,
    model_dim=172,
    encoder_convnext_expansion_dim=216,
    decoder_convnext_expansion_dim=216,
    decoder_posnet_groups_first=(1, 1),
    decoder_posnet_groups_last=(2, 2),
)
OUTPUT = RECIPE_DIR / "LRAC2-Track1-VoCodec-Model-Design-Sheet.xlsx"


def main():
    results = create_workbook(VARIANT, OUTPUT)
    print(f"Created {OUTPUT}")
    for name, value in results.items():
        print(f"{VARIANT.label} {name}: {value}")


if __name__ == "__main__":
    main()
