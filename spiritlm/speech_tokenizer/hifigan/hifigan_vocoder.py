# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Standalone Hifigan vocoder
# Adapted from:
# - https://github.com/jik876/hifi-gan
# - https://github.com/facebookresearch/fairseq/tree/main/fairseq/models/text_to_speech
# - https://github.com/facebookresearch/speech-resynthesis/blob/main/examples/speech_to_speech_translation/models.py
# - https://github.com/facebookresearch/speech-resynthesis/blob/main/examples/expresso/models.py

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

logger = logging.getLogger(__name__)


class HifiGANVocoder(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        config_path=None,
        default_speaker=0,
        default_style=0,
        fp16=False,
    ):
        super().__init__()

        if config_path is None:
            config_path = Path(checkpoint_path).parent / "config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoderModel(checkpoint_path, cfg, fp16)
        self.vocoder.eval()

        self.multispkr = self.vocoder.model.multispkr
        if self.multispkr:
            self.default_speaker = default_speaker
            speakers_path = Path(checkpoint_path).parent / "speakers.txt"
            if speakers_path.exists():
                with open(speakers_path) as f:
                    self.speakers = [line.strip() for line in f]
                print(
                    f"Loaded {len(self.speakers)} speakers. First few speakers: {self.speakers[:10]}"
                )

        self.multistyle = self.vocoder.model.multistyle
        if self.multistyle:
            self.default_style = default_style
            styles_path = Path(checkpoint_path).parent / "styles.txt"
            if styles_path.exists():
                with open(styles_path) as f:
                    self.styles = [line.strip() for line in f]
                print(
                    f"Loaded {len(self.styles)} styles. First few styles: {self.styles[:10]}"
                )

        self.dur_pred = self.vocoder.model.dur_predictor is not None
        self.cfg = cfg

        logger.info(
            f"HifiGAN: Duration Prediction = {self.dur_pred} - "
            f"Multiple Speaker = {bool(self.multispkr)} - "
            f"Multiple Style = {bool(self.multistyle)}"
        )

        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def preprocess_code(self, code, deduplicate_code=False):
        if isinstance(code, str):
            code = code.split()
        if isinstance(code, list):
            code = list(map(int, code))
            code = torch.tensor(code)
        elif isinstance(code, np.ndarray):
            code = torch.from_numpy(code)
        code = code.long()
        if deduplicate_code:
            code = torch.unique_consecutive(code)
        return code.view(1, -1)

    def forward(
        self,
        code,
        speaker_id=None,
        style_id=None,
        dur_pred=True,
        f0_code=None,
        style_code=None,
        not_dedup_code=False,
    ):
        assert not (
            dur_pred and not self.dur_pred
        ), "Model doesnt't support duration prediction"
        inp = dict()
        inp["code"] = self.preprocess_code(code, dur_pred and not not_dedup_code)
        if f0_code is not None:
            inp["f0_code"] = self.preprocess_code(f0_code, deduplicate_code=False)
        if style_code is not None:
            inp["style_code"] = self.preprocess_code(style_code, deduplicate_code=False)
        if self.multispkr:
            if speaker_id is None:
                speaker_id = self.default_speaker
            inp["spkr"] = torch.LongTensor([speaker_id]).view(1, 1)
        if self.multistyle:
            if style_id is None:
                style_id = self.default_style
            inp["style"] = torch.LongTensor([style_id]).view(1, 1)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        return self.vocoder(inp, dur_pred)


class CodeHiFiGANVocoderModel(nn.Module):
    def __init__(
        self, checkpoint_path: str, model_cfg: Dict[str, str], fp16: bool = False
    ) -> None:
        super().__init__()
        self.model = CodeGenerator(model_cfg)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict["generator"])
        self.model.eval()
        if fp16:
            self.model.half()
        self.model.remove_weight_norm()
        print(f"Loaded CodeHiFiGAN checkpoint from '{checkpoint_path}'")

    def upsample(self, code, downsampled_code, uprate):
        N = code.size(1)
        K = downsampled_code.size(1)
        assert abs(K * uprate - N) / uprate <= 1, (N, K, uprate)
        upsampled_code = torch.repeat_interleave(downsampled_code, uprate, dim=1)
        if upsampled_code.size(1) < N:
            z = torch.zeros_like(code)
            z[:, : upsampled_code.size(1)] = upsampled_code
            z[:, upsampled_code.size(1) :] = upsampled_code[:, -1].view(-1, 1)
            upsampled_code = z
        upsampled_code = upsampled_code[:, :N]
        return upsampled_code

    def forward(self, x: Dict[str, torch.Tensor], dur_prediction=False) -> torch.Tensor:
        assert "code" in x
        x["dur_prediction"] = dur_prediction

        # remove invalid code
        mask = x["code"] >= 0
        x["code"] = x["code"][mask].unsqueeze(dim=0)
        if "f0" in x:
            f0_up_ratio = x["f0"].size(1) // x["code"].size(1)
            mask = mask.unsqueeze(2).repeat(1, 1, f0_up_ratio).view(-1, x["f0"].size(1))
            x["f0"] = x["f0"][mask].unsqueeze(dim=0)

        # preprocess f0 & style codes
        if "f0_code" in x:
            if dur_prediction:  # f0 must be upsampled first if dedup
                assert len(x["f0_code"][0]) == len(
                    x["code"][0]
                ), f"f0 must be upsampled first if dedup (f0_code length: {len(x['f0_code'][0])}, code length: {len(x['code'][0])})"
            else:
                x["f0_code"] = self.upsample(
                    x["code"], x["f0_code"], self.model.hubert_to_f0
                )

        if "style_code" in x:
            if dur_prediction:  # style must be upsampled first if dedup
                f"style must be upsampled first if dedup (style_code length: {len(x['style_code'][0])}, code length: {len(x['code'][0])})"
            else:
                x["style_code"] = self.upsample(
                    x["code"], x["style_code"], self.model.hubert_to_style
                )

        return self.model(**x).detach().squeeze()


# Higigan Generator
LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.num_kernels = len(cfg["resblock_kernel_sizes"])
        self.num_upsamples = len(cfg["upsample_rates"])
        self.conv_pre = weight_norm(
            Conv1d(
                cfg.get("model_in_dim", 80),
                cfg["upsample_initial_channel"],
                7,
                1,
                padding=3,
            )
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"])
        ):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        cfg["upsample_initial_channel"] // (2**i),
                        cfg["upsample_initial_channel"] // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = cfg["upsample_initial_channel"] // (2 ** (i + 1))
            for k, d in zip(
                cfg["resblock_kernel_sizes"], cfg["resblock_dilation_sizes"]
            ):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class VariancePredictor(nn.Module):
    def __init__(
        self,
        encoder_embed_dim,
        var_pred_hidden_dim,
        var_pred_kernel_size,
        var_pred_dropout,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                encoder_embed_dim,
                var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size,
                padding=(var_pred_kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.ln1 = nn.LayerNorm(var_pred_hidden_dim)
        self.dropout = var_pred_dropout
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                var_pred_hidden_dim,
                var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.ln2 = nn.LayerNorm(var_pred_hidden_dim)
        self.proj = nn.Linear(var_pred_hidden_dim, 1)

    def forward(self, x):
        # Input: B x T x C; Output: B x T
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(self.ln1(x), p=self.dropout, training=self.training)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(self.ln2(x), p=self.dropout, training=self.training)
        return self.proj(x).squeeze(dim=2)


class CodeGenerator(Generator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dict = nn.Embedding(cfg["num_embeddings"], cfg["embedding_dim"])
        self.multispkr = cfg.get("multispkr", None)
        self.embedder = cfg.get("embedder_params", None)

        self.f0_dict = None
        if cfg.get("num_f0_tokens", None):
            self.f0_dict = nn.Embedding(cfg["num_f0_tokens"], cfg["embedding_dim"])
            self.hubert_to_f0 = round(
                cfg["f0_hop_size"] / cfg["code_hop_size"]
            )  # 4 for 25hz hubert and 6.25hz f0

        self.style_dict = None
        if cfg.get("num_style_tokens", None):
            self.style_dict = nn.Embedding(
                cfg["num_style_tokens"], cfg["embedding_dim"]
            )
            self.hubert_to_style = round(
                cfg["style_hop_size"] / cfg["code_hop_size"]
            )  # 25 for 25hz hubert and 1hz style

        self.multistyle = cfg.get("multistyle", None)

        if self.multispkr and not self.embedder:
            self.spkr = nn.Embedding(cfg.get("num_speakers", 200), cfg["embedding_dim"])
        elif self.embedder:
            self.spkr = nn.Linear(cfg.get("embedder_dim", 256), cfg["embedding_dim"])

        if self.multistyle:
            self.style = nn.Embedding(cfg.get("num_styles", 100), cfg["embedding_dim"])

        self.dur_predictor = None
        if cfg.get("dur_predictor_params", None):
            self.dur_predictor = VariancePredictor(
                cfg["dur_predictor_params"]["encoder_embed_dim"],
                cfg["dur_predictor_params"]["var_pred_hidden_dim"],
                cfg["dur_predictor_params"]["var_pred_kernel_size"],
                cfg["dur_predictor_params"]["var_pred_dropout"],
            )

        self.f0 = cfg.get("f0", None)
        n_f0_bin = cfg.get("f0_quant_num_bin", 0)
        self.f0_quant_embed = (
            None if n_f0_bin <= 0 else nn.Embedding(n_f0_bin, cfg["embedding_dim"])
        )

    @staticmethod
    def _upsample(signal, max_frames):
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError(
                "Padding condition signal - misalignment between condition features."
            )

        signal = signal.view(bsz, channels, max_frames)
        return signal

    def forward(self, **kwargs):
        x = self.dict(kwargs["code"]).transpose(1, 2)

        dur_out = None
        if self.dur_predictor and kwargs.get("dur_prediction", False):
            assert x.size(0) == 1, "only support single sample"
            log_dur_pred = self.dur_predictor(x.transpose(1, 2))
            dur_out = torch.clamp(
                torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1
            )
            # B x C x T
            x = torch.repeat_interleave(x, dur_out.view(-1), dim=2)

        if self.f0:
            if self.f0_quant_embed:
                kwargs["f0"] = self.f0_quant_embed(kwargs["f0"].long()).transpose(1, 2)
            else:
                kwargs["f0"] = kwargs["f0"].unsqueeze(1)

            if x.shape[-1] < kwargs["f0"].shape[-1]:
                x = self._upsample(x, kwargs["f0"].shape[-1])
            elif x.shape[-1] > kwargs["f0"].shape[-1]:
                kwargs["f0"] = self._upsample(kwargs["f0"], x.shape[-1])
            x = torch.cat([x, kwargs["f0"]], dim=1)

        if self.f0_dict is not None:
            f0 = self.f0_dict(kwargs["f0_code"]).transpose(1, 2)  # B, C, T
            if dur_out is not None:
                f0 = torch.repeat_interleave(f0, dur_out.view(-1), dim=2)
            x = torch.cat([x, f0], dim=1)  # B, 2C, T

        if self.style_dict is not None:
            style = self.style_dict(kwargs["style_code"]).transpose(1, 2)  # B, C, T
            if dur_out is not None:
                style = torch.repeat_interleave(style, dur_out.view(-1), dim=2)
            x = torch.cat([x, style], dim=1)  # B, 2C, T

        if self.multispkr:
            assert (
                "spkr" in kwargs
            ), 'require "spkr" input for multispeaker CodeHiFiGAN vocoder'
            spkr = self.spkr(kwargs["spkr"]).transpose(1, 2)
            spkr = self._upsample(spkr, x.shape[-1])
            x = torch.cat([x, spkr], dim=1)

        if self.multistyle:
            assert (
                "style" in kwargs
            ), 'require "style" input for multispeaker CodeHiFiGAN vocoder'
            style = self.style(kwargs["style"]).transpose(1, 2)
            style = self._upsample(style, x.shape[-1])
            x = torch.cat([x, style], dim=1)

        for k, feat in kwargs.items():
            if k in [
                "spkr",
                "code",
                "f0",
                "dur_prediction",
                "style",
                "f0_code",
                "style_code",
            ]:
                continue

            feat = self._upsample(feat, x.shape[-1])
            x = torch.cat([x, feat], dim=1)

        return super().forward(x)
