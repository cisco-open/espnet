# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""VoCodec GAN model for the LRAC 2 Track 1 baseline."""

from typing import Any, Dict, Optional

import torch

from espnet2.gan_codec.abs_gan_codec import AbsGANCodec
from espnet2.gan_codec.encodec.encodec import EncodecDiscriminator
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_codec.shared.loss.loss_balancer import Balancer
from espnet2.gan_codec.soundstream.soundstream import SoundStream
from espnet2.gan_codec.vocodec.generator import VoCodecGenerator
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)


class VoCodec(SoundStream):
    """VoCodec generator trained with an EnCodec-style MS-STFT discriminator."""

    @staticmethod
    def _build_generator(
        sampling_rate: int, generator_params: Dict[str, Any]
    ) -> torch.nn.Module:
        """Build the generator while allowing loss-compatible LRAC 2 variants."""
        parameters = dict(generator_params)
        parameters["sample_rate"] = sampling_rate
        return VoCodecGenerator(**parameters)

    def __init__(
        self,
        apply_enhancement: bool = False,
        sampling_rate: int = 24000,
        generator_params: Optional[Dict[str, Any]] = None,
        discriminator_params: Optional[Dict[str, Any]] = None,
        generator_adv_loss_params: Optional[Dict[str, Any]] = None,
        discriminator_adv_loss_params: Optional[Dict[str, Any]] = None,
        use_feat_match_loss: bool = True,
        feat_match_loss_params: Optional[Dict[str, Any]] = None,
        use_mel_loss: bool = True,
        mel_loss_params: Optional[Dict[str, Any]] = None,
        use_dual_decoder: bool = False,
        lambda_quantization: float = 1.0,
        lambda_reconstruct: float = 0.0,
        lambda_commit: float = 0.25,
        lambda_adv: float = 2.0,
        lambda_feat_match: float = 2.0,
        lambda_mel: float = 15.0,
        cache_generator_outputs: bool = True,
        use_loss_balancer: bool = False,
        balance_ema_decay: float = 0.99,
    ):
        # SoundStream's optimization code is reusable, but its constructor would
        # instantiate an unrelated SEANet generator. Initialize the common module
        # base directly and register only the VoCodec components.
        AbsGANCodec.__init__(self)
        if use_dual_decoder:
            raise ValueError("VoCodec does not support a dual decoder")

        generator_params = dict(generator_params or {})
        discriminator_params = discriminator_params or {
            "msstft_discriminator_params": {
                "filters": 32,
                "in_channels": 1,
                "out_channels": 1,
                "sep_channels": False,
                "n_ffts": [128, 256, 512, 1024, 2048],
                "hop_lengths": [32, 64, 128, 256, 512],
                "win_lengths": [128, 256, 512, 1024, 2048],
                "max_filters": 1024,
                "filters_scale": 1,
                "kernel_size": [3, 9],
                "dilations": [1, 2, 4],
                "stride": [1, 2],
                "normalized": True,
                "norm": "weight_norm",
                "activation": "LeakyReLU",
                "activation_params": {"negative_slope": 0.2},
            }
        }
        generator_adv_loss_params = generator_adv_loss_params or {
            "average_by_discriminators": False,
            "loss_type": "mse",
        }
        discriminator_adv_loss_params = discriminator_adv_loss_params or {
            "average_by_discriminators": False,
            "loss_type": "mse",
        }
        feat_match_loss_params = feat_match_loss_params or {
            "average_by_discriminators": False,
            "average_by_layers": False,
            "include_final_outputs": True,
        }
        mel_loss_params = dict(
            mel_loss_params
            or {
                "range_start": 5,
                "range_end": 12,
                "window": "hann",
                "n_mels": [5, 10, 20, 40, 80, 160, 320],
                "fmin": 0,
                "fmax": None,
                "log_base": None,
                "alphas": False,
                "use_mse": False,
                "scale_reduction": "sum",
            }
        )
        mel_loss_params["fs"] = sampling_rate

        self.apply_enhancement = apply_enhancement
        self.generator = self._build_generator(sampling_rate, generator_params)
        self.discriminator = EncodecDiscriminator(**discriminator_params)
        self.generator_adv_loss = GeneratorAdversarialLoss(**generator_adv_loss_params)
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            **discriminator_adv_loss_params
        )
        self.generator_reconstruct_loss = torch.nn.L1Loss(reduction="mean")

        self.use_feat_match_loss = use_feat_match_loss
        if use_feat_match_loss:
            self.feat_match_loss = FeatureMatchLoss(**feat_match_loss_params)
        self.use_mel_loss = use_mel_loss
        if use_mel_loss:
            self.mel_loss = MultiScaleMelSpectrogramLoss(**mel_loss_params)
        self.use_dual_decoder = False
        self.use_semantic_loss = False

        self.lambda_quantization = lambda_quantization
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_commit = lambda_commit
        self.lambda_adv = lambda_adv
        self.lambda_feat_match = lambda_feat_match
        self.lambda_mel = lambda_mel
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        self.fs = sampling_rate
        self.num_streams = self.generator.quantizer.n_q
        self.frame_shift = self.generator.frame_shift
        self.code_size_per_stream = [self.generator.quantizer.bins] * self.num_streams
        self.loss_balancer = (
            Balancer(ema_decay=balance_ema_decay, per_batch_item=True)
            if use_loss_balancer
            else None
        )
