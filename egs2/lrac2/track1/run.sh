#!/usr/bin/env bash
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0


set -e
set -u
set -o pipefail

fs=24000

train_set=train
valid_set=train_validation
test_sets="evaluation \
open-evaluation/open_testset_track1_clean \
open-evaluation/open_testset_track1_noisy \
open-evaluation/open_testset_track1_reverb \
test-clean \
test-realworld \
test-simultaneous-talkers"

train_config=conf/train.yaml
inference_config=conf/decode.yaml
score_config=conf/score.yaml
target_bandwidths="1 6 12 24 32"

./codec.sh \
    --fs "${fs}" \
    --audio_format wav \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --scoring_config "${score_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --target_bandwidths "${target_bandwidths}" \
    "$@"
