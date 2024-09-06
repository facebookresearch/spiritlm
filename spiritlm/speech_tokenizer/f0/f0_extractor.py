# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
import torchaudio


class F0Extractor(nn.Module):

    def __init__(
        self,
        hop_length=80,
        sampling_rate=16000,
        interpolate=True,
    ):
        """Each second will have sampling_rate/hop_length frames."""
        super().__init__()

        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.interpolate = interpolate

    def load_audio(self, path, mono=True):
        wav, sr = torchaudio.load(path)
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=self.sampling_rate
            )
        if mono and wav.ndim == 2:
            wav = wav.mean(dim=0)
        wav = wav.numpy()
        return wav

    def compute_f0_uv(self, wav, interpolate=True):
        raise NotImplementedError("Not implemented!")

    @torch.inference_mode()
    def forward(self, audio, vuv=False):
        if isinstance(audio, str):
            audio = self.load_audio(audio)

        f0, uv = self.compute_f0_uv(audio, interpolate=self.interpolate)

        if not vuv:
            return f0
        else:
            return f0, uv


class pYAAPTF0Extractor(F0Extractor):

    def compute_f0_uv(self, wav, interpolate=True):
        pitch = self.get_pitch(wav)
        # take interpolate, otherwise pitch.samp_values
        # pyaapt has some problems with pitch.samp_values, so do it manually (from pgslm)
        f0 = pitch.samp_values
        if interpolate:
            f0 = self.interpolate_f0(f0)
        vuv = pitch.vuv
        return f0, vuv

    def get_pitch(self, wav):
        try:
            import amfm_decompy.basic_tools as basic
            import amfm_decompy.pYAAPT as pYAAPT
            from librosa.util import normalize
        except ImportError as error:
            raise ImportError(
                "To use pYAAPTF0Extractor, please install AMFM-decompy and librosa"
            ) from error

        wav = wav.squeeze()
        assert wav.ndim == 1
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav)
        frame_length = 20.0  # ms
        to_pad = int(frame_length / 1000 * self.sampling_rate) // 2

        # remove remainders for large hop length
        n_frames = len(wav) // self.hop_length * self.hop_length
        wav = wav[:n_frames]

        audio = normalize(wav) * 0.95
        if self.hop_length == 80:
            audio = np.pad(audio, (to_pad, to_pad), "constant", constant_values=0)
        audio = basic.SignalObj(audio, self.sampling_rate)
        pitch = pYAAPT.yaapt(
            audio,
            frame_length=frame_length,
            frame_space=self.hop_length / self.sampling_rate * 1000,
            nccf_thresh1=0.25,
            tda_frame_length=25.0,
        )

        return pitch

    def interpolate_f0(self, f0, fill_extremities=True):
        try:
            from scipy.interpolate import interp1d
        except ImportError as error:
            raise ImportError(
                "To use pYAAPTF0Extractor, please install scipy (`pip install scipy`)"
            ) from error

        orig_t = np.arange(f0.shape[0])
        f0_interp = f0[:]
        ii = f0_interp != 0
        if ii.sum() > 1:
            f0_interp = interp1d(
                orig_t[ii],
                f0_interp[ii],
                bounds_error=False,
                kind="linear",
                fill_value=0,
            )(orig_t)

            # Fill extreme values with border values
            if fill_extremities:
                f0_interp[: orig_t[ii][0]] = f0_interp[ii][0]
                f0_interp[orig_t[ii][-1] + 1 :] = f0_interp[ii][-1]

        return f0_interp


class FCPEF0Extractor(F0Extractor):

    def __init__(
        self,
        hop_length=80,
        sampling_rate=16000,
        interpolate=True,
        device=None,
    ):
        try:
            from torchfcpe import spawn_bundled_infer_model
        except ImportError as error:
            raise ImportError(
                "To use FCPEF0Extractor, please install torchfcpe (`pip install torchfcpe`)"
            ) from error

        super().__init__(
            hop_length=hop_length, sampling_rate=sampling_rate, interpolate=interpolate
        )

        self.model = spawn_bundled_infer_model(device=device)

    def compute_f0_uv(self, wav, interpolate=True):
        wav = wav.squeeze()
        assert wav.ndim == 1
        f0_target_length = (len(wav) // self.hop_length) + 1
        if not isinstance(wav, torch.Tensor):
            wav = torch.from_numpy(wav)
        wav = wav.float().unsqueeze(0).unsqueeze(-1)
        f0, uv = self.model.infer(
            wav,
            sr=self.sampling_rate,
            decoder_mode="local_argmax",  # Recommended mode
            threshold=0.05,  # Threshold for V/UV decision
            f0_min=50,  # Minimum pitch
            f0_max=1100,  # Maximum pitch
            interp_uv=interpolate,  # Interpolate unvoiced frames
            output_interp_target_length=f0_target_length,  # Interpolate to target length
            retur_uv=True,
        )
        vuv = 1 - uv
        return f0.squeeze().cpu().numpy(), vuv.squeeze().cpu().numpy()


def load_f0_extractor(
    f0_extractor_method, hop_length, sampling_rate, interpolate, device=None
):
    expected_methods = ["pyaapt", "fcpe"]
    assert (
        f0_extractor_method in expected_methods
    ), f"Unexpected f0 extractor method: {f0_extractor_method} (choices are: {expected_methods})"
    if f0_extractor_method == "pyaapt":
        f0_extractor = pYAAPTF0Extractor(
            hop_length=hop_length, sampling_rate=sampling_rate, interpolate=interpolate
        )
    elif f0_extractor_method == "fcpe":
        f0_extractor = FCPEF0Extractor(
            hop_length=hop_length,
            sampling_rate=sampling_rate,
            interpolate=interpolate,
            device=device,
        )
    print(
        f"Using '{f0_extractor_method}' f0 extractor method (choices are: {expected_methods})"
    )
    return f0_extractor
