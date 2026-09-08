#!/usr/bin/env python3
"""Generate and verify the final LRAC 2 Track 2 VoCodec design workbook."""

import sys
from pathlib import Path


RECIPE_DIR = Path(__file__).resolve().parents[1]
LRAC2_COMMON_DIR = RECIPE_DIR.parent / "common"
sys.path.insert(0, str(LRAC2_COMMON_DIR))

from vocodec_design_sheet import Variant, create_workbook  # noqa: E402


VARIANT = Variant(
    track=2,
    label="100 Hz / 32-layer / 1.5M cap",
    hop_size=240,
    attention_context_frames=300,
    attention_lookahead_frames=0,
    attention_heads=4,
    rvq_layers=32,
    model_dim=96,
    encoder_convnext_expansion_dim=128,
    decoder_convnext_expansion_dim=128,
    decoder_posnet_groups_first=(1, 1),
    decoder_posnet_groups_last=(1, 1),
    parameter_budget=1_500_000,
)
OUTPUT = RECIPE_DIR / "LRAC2-Track2-VoCodec-Model-Design-Sheet.xlsx"


def main():
    results = create_workbook(VARIANT, OUTPUT)
    print(f"Created {OUTPUT}")
    for name, value in results.items():
        print(f"{VARIANT.label} {name}: {value}")


if __name__ == "__main__":
    main()
