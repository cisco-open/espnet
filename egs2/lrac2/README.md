# Low-Resource Audio Codec Challenge 2.0 (LRAC 2.0) Baseline Recipe

This directory contains the ESPnet baseline recipes for the second edition of
the [Low-Resource Audio Codec Challenge (LRAC 2.0)](https://crowdsourcing.cisco.com/lrac-challenge/2026/).
The challenge studies low-complexity, low-bitrate speech codecs under strict
compute, bitrate, and latency constraints.

LRAC 2.0 has two tracks:

- **Track 1 — Transparency:** optimize reconstructed speech quality under the
  common challenge constraints.
- **Track 2 — Memory-Efficient Transparency:** solve the same transparency task
  with an additional 1.5-million-parameter cap and a memory-bandwidth penalty.

See the official [track descriptions](https://crowdsourcing.cisco.com/lrac-challenge/2026/tracks)
for the complete definitions and submission rules.

The current baseline is adapted from VoCodec [1]. The recipes, configurations,
implementation, validation tools, and model-design workbooks are included here.

## Baseline checkpoints

The final trained baseline checkpoints are available for both tracks:

- [Track 1 baseline checkpoint](track1/baseline_model/LRAC2026_Track1_baseline_model.pth)
- [Track 2 baseline checkpoint](track2/baseline_model/LRAC2026_Track2_baseline_model.pth)

Both checkpoints contain the model state saved after epoch 1700. The model
weights are distributed under the Creative Commons Attribution-NonCommercial
4.0 International license; see the license files accompanying the
[Track 1](track1/baseline_model/LICENSE.txt) and
[Track 2](track2/baseline_model/LICENSE.txt) checkpoints. A paper describing
the LRAC 2.0 baseline system in greater detail will be added in a future update.

## Getting started

Each track has its own recipe under [`track1`](track1) or [`track2`](track2).
Before running either recipe:

1. Set up ESPnet and the dependencies required by the metrics in
   [`conf/score.yaml`](track1/conf/score.yaml).
2. Download and prepare the LRAC 2.0 data using the
   [LRAC data-generation toolkit](https://github.com/cisco-open/lrac_data_generation).
3. Copy or symbolically link the generated ESPnet `data` directory into each
   track directory.

For example:

```bash
cd egs2/lrac2/track1
ln -s /path/to/lrac-espnet-data data
./run.sh --stage 1 --stop_stage 1
```

Use the corresponding directory for Track 2. The data directory must contain
the `train`, `train_validation`, and evaluation sets named in `run.sh`.

For a general introduction to ESPnet recipes, see:

- [ESPnet2 tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [Speech codec recipe](https://espnet.github.io/espnet/recipe/codec1.html)
- [ESPnet recipe templates](https://espnet.github.io/espnet/recipe/)

## Recipe stages

Each track uses the following seven stages:

1. **Data checks:** verify that the data directories produced by the LRAC
   data-generation repository are present.
2. **Formatting:** format `wav.scp` and `reference.scp` and write the results
   under `dump/raw`.
3. **Data filtering:** remove training and validation recordings outside the
   duration limits defined in `codec.sh`. Evaluation sets are not filtered.
4. **Statistics collection:** skipped because the codec operates directly on
   raw audio.
5. **Training:** train the codec and validate it after every epoch, retaining
   checkpoints according to the criteria in the training configuration.
6. **Inference:** reconstruct the evaluation sets at each requested bitrate.
7. **Scoring:** evaluate the reconstructed audio using the metrics in
   `conf/score.yaml`.

Run a single stage with, for example:

```bash
cd egs2/lrac2/track1
./run.sh --stage 5 --stop_stage 5
```

Although `codec.sh` can run all stages in sequence, running the stages one at a
time is usually easier to monitor and debug.

## Important recipe files

- [`track1/run.sh`](track1/run.sh) and [`track2/run.sh`](track2/run.sh): define
  the datasets, configurations, and target bitrates, and pass command-line
  options to `codec.sh`.
- [`track1/codec.sh`](track1/codec.sh) and [`track2/codec.sh`](track2/codec.sh):
  implement the seven recipe stages.
- [`track1/conf/train.yaml`](track1/conf/train.yaml) and
  [`track2/conf/train.yaml`](track2/conf/train.yaml): define the model, losses,
  optimization, and data-loading settings.
- [`track1/conf/decode.yaml`](track1/conf/decode.yaml) and
  [`track2/conf/decode.yaml`](track2/conf/decode.yaml): define inference.
- [`track1/conf/score.yaml`](track1/conf/score.yaml) and
  [`track2/conf/score.yaml`](track2/conf/score.yaml): select the objective
  metrics computed through ESPnet-VERSA.
- `local/path.sh`: adds the ESPnet source tree to `PYTHONPATH`.

## Baseline model

### VoCodec architecture

VoCodec is a lightweight time-frequency neural speech codec. Its analysis path
converts 24-kHz audio to an STFT-domain representation using a 720-sample
(30-ms) window and a 240-sample (10-ms) hop. Causal ConvNeXt blocks and a
PosNet module containing residual convolutions and self-attention produce a
100-Hz latent representation. A residual vector quantizer (RVQ) compresses the
latent sequence. The decoder mirrors the neural processing and uses an inverse
STFT to reconstruct the waveform. Training uses a multi-scale STFT
discriminator together with adversarial, feature-matching, multi-scale mel,
commitment, and codebook losses.

The LRAC 2.0 implementation retains the main VoCodec design while adapting it
to the new track definitions:

- The RVQ depth is increased from the original paper's six layers to 24 layers.
  With 1,024-entry codebooks and a 100-Hz frame rate, each layer contributes
  1 kbps. The model therefore supports native 1, 6, 12, and 24 kbps operating
  points. A 32-kbps inference request uses all 24 available layers and is
  therefore equivalent to the 24-kbps operating point.
- Both tracks use four-head bounded multi-head self-attention with a 300-frame
  window (approximately three seconds at 100 Hz), zero attention lookahead,
  and rotary position embeddings (RoPE).
- Track 1 uses a 176-dimensional model. Its encoder and decoder ConvNeXt
  expansion dimensions are 216 and 200, respectively.
- Track 2 uses a 112-dimensional model and a 128-dimensional expansion in both
  the encoder and decoder so that the deployed model remains below the
  1.5-million-parameter cap.
- In both tracks, the decoder PosNet residual groups are `[1, 1]`, `[1, 1]`,
  `[2, 2]`, and `[2, 2]` for blocks one through four.

The query tile size in the attention configuration controls only the size of a
GPU work tile. It does not change the temporal context, add lookahead, or alter
algorithmic latency.

### Current baseline configurations

| Setting | Track 1 | Track 2 |
| --- | ---: | ---: |
| Model dimension | 176 | 112 |
| Encoder / decoder expansion | 216 / 200 | 128 / 128 |
| Attention heads | 4 | 4 |
| Attention context | 300 frames | 300 frames |
| RoPE rotary dimension | 44 | 28 |
| RVQ layers | 24 | 24 |
| Native target bitrates | 1, 6, 12, 24 kbps | 1, 6, 12, 24 kbps |
| Generator trainable parameters | 3,179,362 | 1,478,626 |
| Cap-counted persistent values | 3,180,802 | 1,480,066 |

The analytical model-design calculation gives the following constraint values:

| Quantity | Track 1 | Track 2 |
| --- | ---: | ---: |
| Transmit compute | 384.286 MFLOPS | 190.748 MFLOPS |
| Receive compute | 294.276 MFLOPS | 132.073 MFLOPS |
| Total latency (buffering + algorithmic) | 30 ms | 30 ms |
| Memory bandwidth | 694.883 MB/s | 333.536 MB/s |
| Track 2 bandwidth penalty | Not applicable | 9.502 |

Under the assumptions encoded by the design-sheet generator, Track 1 remains
within the 400-MFLOPS transmit, 300-MFLOPS receive, and 30-ms latency limits.
Track 2 meets the same limits and its additional 1.5-million-parameter cap.
The challenge rules, rather than this summary, are authoritative.

## DAC-style residual vector quantizer

The VoCodec baseline uses the new
[`DACResidualVectorQuantizer`](../../espnet2/gan_codec/shared/quantizer/dac_residual_vq.py)
instead of ESPnet's existing
[`ResidualVectorQuantizer`](../../espnet2/gan_codec/shared/quantizer/residual_vq.py).
The existing implementation follows the EnCodec/vector-quantize-pytorch design
and includes EMA codebook updates, k-means initialization, and dead-code
replacement. The new implementation is provided because VoCodec uses a
DAC-style quantizer with different training and projection behavior:

- codebook embeddings are optimized through an explicit codebook loss, while a
  straight-through estimator and commitment loss train the encoder path;
- nearest-neighbor lookup can use normalized encoder and codebook vectors while
  reconstruction uses the selected unnormalized embedding;
- projection can be applied independently around every RVQ layer
  (`per_layer`, used by the baselines), once around the complete stack
  (`shared`), or omitted when the dimensions match (`none`);
- inference and `encode`/`decode` use deterministic RVQ prefixes;
- training supports per-example RVQ-prefix masks and keeps every RVQ layer in
  the training graph, including a layer that is inactive for a local batch.

This is a separate implementation rather than a replacement for the existing
ESPnet RVQ. Codecs that depend on the existing EMA behavior continue to use the
existing class.

### Quantizer dropout

The current recipes use:

```yaml
quantizer_n_q: 24
quantizer_bins: 1024
quantizer_dropout: 0.5
quantizer_dropout_sampling: target_bandwidths
quantizer_target_bandwidth: [1, 6, 12, 24]
```

`quantizer_dropout` is the fraction of examples in each training batch selected
for random prefix sampling. The remaining examples use all 24 layers. For a
selected example, `quantizer_dropout_sampling: target_bandwidths` samples
uniformly from the RVQ prefixes corresponding to
`quantizer_target_bandwidth`—here 1, 6, 12, or 24 layers. It does **not** sample
the intermediate layer counts. At the current 100-Hz frame rate and 1,024-entry
codebooks, these prefix lengths correspond exactly to 1, 6, 12, and 24 kbps.

For example, with a sufficiently large batch and `quantizer_dropout: 0.5`, the
expected proportions are 12.5% each at 1, 6, and 12 layers, and 62.5% at all 24
layers: half of the examples are forced to the full stack and the sampled half
chooses uniformly among the four targets. The implementation selects
`int(batch_size * quantizer_dropout)` examples, so small local batches may have
coarser proportions.

To sample every selected prefix from all layer counts `1..24` instead, use:

```yaml
quantizer_dropout: 0.5
quantizer_dropout_sampling: all_layers
```

Useful special cases are:

- `quantizer_dropout: 0.0`: disable random prefix sampling; every training
  example uses all layers.
- `quantizer_dropout: 1.0` with `target_bandwidths`: every training example
  samples uniformly from the configured target operating points.
- an explicit bandwidth or quantizer count in the quantizer API: use that
  prefix and bypass random dropout sampling.

During evaluation, dropout sampling is disabled. If no bitrate is requested,
the generator defaults to the last value in `quantizer_target_bandwidth`.
Target bandwidths must be positive, distinct, exactly representable by the
configured frame rate and codebook size, and within the available RVQ depth.

## Streaming design and streamability checker

All temporal convolutions in the VoCodec path are causal. Self-attention is
limited to a 300-frame sliding window with zero lookahead, so its memory and
compute do not grow with utterance duration. RoPE supplies relative positional
information within this window.

The implemented STFT uses centre-aligned hop framing. Let `N` be the window
size, `H` the hop size, and `E = (N - H) / 2`. Frame `t` is assigned to waveform
hop `[tH, (t + 1)H - 1]` and covers
`[tH - E, (t + 1)H + E - 1]`. For `N = 720` and `H = 240`, the 10-ms frame hop
is buffered first, STFT analysis then needs 240 samples (10 ms) beyond the
assigned hop, and the matching inverse STFT waits for one later frame (10 ms)
before finalizing that output hop. This gives 10 ms of buffering plus 20 ms of
algorithmic lookahead. The analysis and synthesis each retain `N - H = 480`
samples of overlap state; this state size is not itself latency.

For comparison, a right-aligned STFT assigns frame `t` to the last hop of its
window. It can produce that frame as soon as the assigned hop is complete, so
there is no additional analysis lookahead beyond the 10-ms hop buffer. Its
matching inverse STFT must instead wait for `N - H = 480` samples (20 ms) of
later frames to complete overlap-add. The two alignments place the delay on
different sides of the codec. The three seconds of left context used by bounded
attention are history and therefore do not add lookahead latency.

[`streamability.py`](../../espnet2/gan_codec/shared/streamability.py) provides a
sample-based check of the complete waveform-to-waveform path. It changes the
input after randomly selected cutoffs and verifies that output earlier than the
allowed future-dependency boundary remains unchanged:

```python
from espnet2.gan_codec.shared.streamability import check_codec_streamability

result = check_codec_streamability(
    codec_or_generator,
    waveform,                         # (batch, channels, samples)
    sample_rate=24000,
    max_future_dependency_ms=30.0,
    num_trials=16,
    atol=1e-5,
    rtol=1e-4,
    seed=0,
)
assert result.passed, result
```

Important limitations and tolerance guidance:

- Passing is empirical evidence only for the tested waveforms, cutoff
  positions, seeds, device, and numerical precision. Repeat the check with
  representative inputs and multiple seeds.
- The default tolerances depend on dtype and device. They are starting points
  for floating-point roundoff, not universal acceptance thresholds. First
  measure repeat-run differences for identical inputs on the target device,
  then set explicit tolerances just above that numerical floor. Do not increase
  tolerances merely to hide a meaningful waveform difference.
- The checker assumes the output length equals the input length minus any
  explicitly declared left crop and that the codec does not resample time.
- The checker tests bounded future dependence. It does not measure execution
  time, packetization, buffering outside the model, scheduling delay, or
  memory use.
- This checker is provided as a helper and may have limitations. Participants
  are responsible for verifying that their submitted codec complies with the
  official challenge rules for latency and streamability.

The checker can be used on a codec wrapper, a generator, or a callable that
returns a waveform. `assert_codec_streamable` provides the same check and
raises an exception on failure.

## Analytical design sheets and latency calculator

[`common/vocodec_design_sheet.py`](common/vocodec_design_sheet.py) is the shared
analytical calculator and Excel-workbook generator. It validates that each
track's `train.yaml` agrees with its declared model variant, calculates
parameters, persistent values, transmit and receive compute, memory bandwidth,
field of view, and algorithmic latency, checks the configured limits, and then
writes the workbook.

From `egs2/lrac2`, regenerate both final workbooks with:

```bash
python track1/local/generate_vocodec_design_sheet.py
python track2/local/generate_vocodec_design_sheet.py
```

The generated deliverables are:

- [LRAC2 Track 1 VoCodec model-design sheet](track1/LRAC2-Track1-VoCodec-Model-Design-Sheet.xlsx)
- [LRAC2 Track 2 VoCodec model-design sheet](track2/LRAC2-Track2-VoCodec-Model-Design-Sheet.xlsx)

These calculations are specific to this VoCodec implementation. The formulas
encode assumptions about its STFT/iSTFT, ConvNeXt and PosNet blocks, attention,
DAC-style RVQ, precision, cached state, and challenge accounting conventions.
They are **not** a general parameter, complexity, latency, or memory-bandwidth
calculator for arbitrary codec architectures. If the implementation or an
accounting assumption changes, update and validate the formulas before relying
on a generated sheet.

The total latency reported in the sheets combines frame-hop buffering and
algorithmic future dependence; it is not measured wall-clock delay. The compute
calculation excludes FFT arithmetic, and the memory-bandwidth calculation
follows the explicit persistent-value and state-traffic assumptions documented
in the workbooks.

## Main implementation files

- [`vocodec/vocodec.py`](../../espnet2/gan_codec/vocodec/vocodec.py): the
  `VoCodec` GAN training wrapper and loss integration.
- [`vocodec/generator.py`](../../espnet2/gan_codec/vocodec/generator.py):
  `STFTAnalysis`, `STFTSynthesis`, `VoCodecEncoder`, `VoCodecDecoder`, and
  `VoCodecGenerator`.
- [`vocodec/blocks.py`](../../espnet2/gan_codec/vocodec/blocks.py):
  `ConvNeXtBlock1d`, `CausalResNetBlock1d`, `BoundedMultiheadAttention`, and
  `PosNet`, including bounded attention and RoPE.
- [`dac_residual_vq.py`](../../espnet2/gan_codec/shared/quantizer/dac_residual_vq.py):
  the trainable DAC-style vector-quantization layer, RVQ stack, and
  bandwidth-aware wrapper.
- [`streamability.py`](../../espnet2/gan_codec/shared/streamability.py): the
  bounded-future-dependency checker.
- [`lrac2/lrac2.py`](../../espnet2/gan_codec/lrac2/lrac2.py): the alternative
  `LRAC2ConvBaseline` convolutional model family. The released Track 1 and
  Track 2 configurations select `codec: vocodec`.
- [`freq_loss.py`](../../espnet2/gan_codec/shared/loss/freq_loss.py): the
  configurable multi-scale mel-spectrogram loss used by the baseline.
- [`gan_codec.py`](../../espnet2/tasks/gan_codec.py): registers `vocodec` and
  `lrac2_conv` with `GANCodecTask`.
- [`gan_codec_train.py`](../../espnet2/bin/gan_codec_train.py): training entry
  point.
- [`gan_codec_inference.py`](../../espnet2/bin/gan_codec_inference.py):
  inference entry point.

## Reference

[1] L. Yang, R. Hu, Y. Xu, and J. Lu, “VoCodec: An Efficient Lightweight
Low-Bitrate Speech Codec,” *2026 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP)*, Barcelona, Spain, 2026, pp. 21982–21986,
doi: [10.1109/ICASSP55912.2026.11464230](https://ieeexplore.ieee.org/document/11464230).
