# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

from .f0_extractor import load_f0_extractor
from .vqvae import load_vqvae


class F0Tokenizer(torch.nn.Module):

    def __init__(
        self,
        f0_extractor_method,
        quantizer_path,
        f0_speaker_stats=None,
        hop_length=80,
        sampling_rate=16000,
        interpolate=False,
        device="cuda",
    ):
        super().__init__()

        self.f0_extractor = load_f0_extractor(
            f0_extractor_method=f0_extractor_method,
            hop_length=hop_length,
            sampling_rate=sampling_rate,
            interpolate=interpolate,
            device=device,
        )

        self.quantizer, self.quantizer_cfg = load_vqvae(quantizer_path)
        self.quantizer.eval()
        self.quantizer.to(device)
        # Load speaker stats
        self.speaker_f0_stats = f0_speaker_stats
        if self.speaker_f0_stats is None and (
            self.quantizer_cfg.get("speaker_norm", False)
            or "norm_" in self.quantizer_cfg.features
        ):
            speaker_stats_path = self.quantizer_cfg.get("speaker_stats", None)
            if speaker_stats_path is not None and os.path.exists(speaker_stats_path):
                self.speaker_f0_stats = torch.load(speaker_stats_path)
                print(f"Speaker f0 stats loaded from '{speaker_stats_path}'")
            else:
                print(
                    "It seems that model is using normalized f0 but no speaker stats is given, will infer mean f0 from input utterance."
                )

        # this is useful for determining the device
        self.register_buffer(
            "_float_tensor", torch.tensor([0], dtype=torch.float, device=device)
        )

    @property
    def device(self):
        return self._float_tensor.device

    def quantize_vqvae(self, f0, vuv, speaker=None, compute_vqvae_pred=False):
        assert self.quantizer_cfg.features in [
            "f0_interp,vuv",
            "f0,vuv",
            "norm_f0_interp,vuv",
            "norm_f0,vuv",
        ], self.quantizer_cfg.features

        if not isinstance(f0, torch.Tensor):
            f0 = torch.tensor(f0)
        if not isinstance(vuv, torch.Tensor):
            vuv = torch.tensor(vuv)

        # normalize f0
        if (
            self.quantizer_cfg.get("speaker_norm", False)
            or "norm_" in self.quantizer_cfg.features
        ):
            mask = f0 != 0
            if speaker is not None and speaker in self.speaker_f0_stats:
                mean = self.speaker_f0_stats[speaker]["f0_mean"]
            else:
                # Get statistics from utterance (maybe it is more accurate to get mean from voiced segments)
                vuv_mask = vuv != 0
                mean = torch.mean(f0[vuv_mask])
            f0[mask] = f0[mask] - mean

        x = torch.stack([f0, vuv])  # (2, T)
        x = x.float().unsqueeze(0).to(self.device)  # (1, 2, T)
        if not compute_vqvae_pred:
            quant_f0 = self.quantizer(x, compute_pred=False)
            quant_f0 = quant_f0[0].squeeze(0).detach().cpu().numpy()
            return quant_f0
        else:
            quant_f0, pred = self.quantizer(x, compute_pred=True)
            quant_f0 = quant_f0[0].squeeze(0).detach().cpu().numpy()
            pred = pred[0].detach().cpu().numpy()
            return quant_f0, pred

    def forward(self, x, speaker=None, dense=False, compute_vqvae_pred=False):
        f0, vuv = self.f0_extractor(x, vuv=True)
        if dense:
            return f0
        return self.quantize_vqvae(
            f0, vuv, speaker=speaker, compute_vqvae_pred=compute_vqvae_pred
        )
