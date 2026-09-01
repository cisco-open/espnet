"""Utilities for checking whether an audio codec is streamable.

The check is sample based and targets the complete waveform-to-waveform path.  A
codec may use a declared amount of future audio, but changing input after a
cutoff must not change output earlier than that allowance.  Running this check
does not require a stateful streaming implementation.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn


@dataclass(frozen=True)
class StreamabilityResult:
    """Result returned by :func:`check_codec_streamability`."""

    passed: bool
    max_absolute_error: float
    max_relative_error: float
    trials: int
    compared_samples: int
    allowed_future_samples: int
    absolute_tolerance: float
    relative_tolerance: float


def _default_tolerances(x: torch.Tensor) -> Tuple[float, float]:
    if x.dtype in (torch.float16, torch.bfloat16):
        return 1e-3, 1e-2
    if x.is_cuda:
        return 1e-5, 1e-4
    return 1e-6, 1e-5


def _waveform_from_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _waveform_from_output(output[0])
    if isinstance(output, dict):
        for key in ("wav", "audio", "audio_hat"):
            if key in output:
                return _waveform_from_output(output[key])
    raise TypeError("The codec callable must return a waveform tensor.")


def _codec_callable(codec_or_callable) -> Callable[[torch.Tensor], torch.Tensor]:
    # ESPnet codec wrappers expose the waveform path through ``generator``;
    # their own forward method returns GAN losses rather than reconstructed audio.
    target = getattr(codec_or_callable, "generator", codec_or_callable)

    def run(x: torch.Tensor) -> torch.Tensor:
        return _waveform_from_output(target(x))

    return run


@torch.no_grad()
def check_codec_streamability(
    codec_or_callable: Union[nn.Module, Callable],
    waveform: torch.Tensor,
    sample_rate: int = 24000,
    max_future_dependency_ms: float = 30.0,
    crop_left_samples: int = 0,
    num_trials: int = 8,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    seed: int = 0,
) -> StreamabilityResult:
    """Test bounded future dependence of a waveform-to-waveform codec.

    Args:
        codec_or_callable: Codec, generator, or callable. Tuple outputs use their
            first item; dictionaries may contain ``wav``, ``audio``, or
            ``audio_hat``.
        waveform: Input shaped ``(batch, channels, samples)``.
        sample_rate: Waveform sampling rate.
        max_future_dependency_ms: Maximum allowed algorithmic lookahead for the
            complete codec, in milliseconds.
        crop_left_samples: Optional startup samples removed from the output. An
            output sample at index zero then corresponds to this input index.
        num_trials: Number of independently located future perturbations.
        atol: Absolute comparison tolerance. Defaults depend on dtype/device and
            are reported in the returned result.
        rtol: Relative comparison tolerance. Defaults depend on dtype/device and
            are reported in the returned result.
        seed: Seed used to choose cutoffs and perturbations.

    The function deliberately works in waveform samples, not latent frames. It
    assumes no temporal resampling: output length is input length minus
    ``crop_left_samples``.

    A sample passes when ``abs(candidate - reference) <= atol + rtol *
    abs(reference)``. The defaults are conservative starting points for normal
    floating-point roundoff, not universal acceptance criteria. For deployment
    validation, first measure repeat-run differences with identical inputs on
    the target device, choose explicit tolerances just above that numerical
    floor, and then repeat this check across representative inputs, seeds, and
    cutoff positions. Tolerances large enough to hide audible or otherwise
    meaningful waveform changes invalidate the test.

    Passing with a 30 ms allowance provides empirical evidence that the tested
    waveform path has no input-dependent lookahead beyond 30 ms. It does not
    measure execution time, packet buffering, scheduling delay, or prove the
    behavior for inputs and cutoff alignments that were not tested.
    """
    if waveform.ndim != 3:
        raise ValueError("waveform must have shape (batch, channels, samples)")
    if sample_rate <= 0 or max_future_dependency_ms < 0:
        raise ValueError("sample_rate must be positive and latency non-negative")
    if crop_left_samples < 0 or num_trials <= 0:
        raise ValueError("crop_left_samples must be non-negative and trials positive")

    allowed = int(round(sample_rate * max_future_dependency_ms / 1000.0))
    minimum_cutoff = allowed + crop_left_samples + 1
    if waveform.shape[-1] <= minimum_cutoff:
        raise ValueError("waveform is too short for the requested latency allowance")

    if atol is None or rtol is None:
        default_atol, default_rtol = _default_tolerances(waveform)
        atol = default_atol if atol is None else atol
        rtol = default_rtol if rtol is None else rtol
    atol = float(atol)
    rtol = float(rtol)
    if not math.isfinite(atol) or not math.isfinite(rtol):
        raise ValueError("atol and rtol must be finite")
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")

    module = codec_or_callable if isinstance(codec_or_callable, nn.Module) else None
    was_training = module.training if module is not None else None
    if module is not None:
        module.eval()

    run = _codec_callable(codec_or_callable)
    try:
        reference = run(waveform)
        if reference.ndim < 1:
            raise ValueError("codec output must have a sample dimension")
        expected_length = waveform.shape[-1] - crop_left_samples
        if reference.shape[-1] != expected_length:
            raise ValueError(
                "codec output length must equal input length minus crop_left_samples"
            )

        generator = torch.Generator(device=waveform.device)
        generator.manual_seed(seed)
        cutoffs = torch.randint(
            minimum_cutoff,
            waveform.shape[-1],
            (num_trials,),
            generator=generator,
            device=waveform.device,
        )

        max_abs = 0.0
        max_rel = 0.0
        compared = 0
        passed = True
        for cutoff_tensor in cutoffs:
            cutoff = int(cutoff_tensor.item())
            perturbed = waveform.clone()
            replacement = torch.randn(
                perturbed[..., cutoff:].shape,
                dtype=perturbed.dtype,
                device=perturbed.device,
                generator=generator,
            )
            perturbed[..., cutoff:] = replacement
            candidate = run(perturbed)
            if candidate.shape != reference.shape:
                raise ValueError("codec output shape changed after perturbation")

            output_stop = cutoff - allowed - crop_left_samples
            ref_prefix = reference[..., :output_stop]
            candidate_prefix = candidate[..., :output_stop]
            error = (candidate_prefix - ref_prefix).abs()
            scale = ref_prefix.abs().clamp_min(float(atol))
            max_abs = max(max_abs, float(error.max().item()))
            max_rel = max(max_rel, float((error / scale).max().item()))
            compared += output_stop
            passed = passed and torch.allclose(
                candidate_prefix, ref_prefix, atol=atol, rtol=rtol
            )

        return StreamabilityResult(
            passed=bool(passed),
            max_absolute_error=max_abs,
            max_relative_error=max_rel,
            trials=num_trials,
            compared_samples=compared,
            allowed_future_samples=allowed,
            absolute_tolerance=float(atol),
            relative_tolerance=float(rtol),
        )
    finally:
        if module is not None:
            module.train(was_training)


def assert_codec_streamable(*args, **kwargs) -> StreamabilityResult:
    """Run :func:`check_codec_streamability` and raise on failure."""
    result = check_codec_streamability(*args, **kwargs)
    if not result.passed:
        raise AssertionError(
            "Codec exceeded its declared future dependency: "
            f"max_abs={result.max_absolute_error:.3e}, "
            f"max_rel={result.max_relative_error:.3e}, "
            f"atol={result.absolute_tolerance:.3e}, "
            f"rtol={result.relative_tolerance:.3e}. "
            "Repeat with representative inputs and explicit tolerances when "
            "validating a deployment model."
        )
    return result
