# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchaudio
from torch import nn

from .hubert_model import load_hubert_model
from .quantizer_model import load_quantizer_model


class HubertTokenizer(nn.Module):
    def __init__(
        self,
        hubert_ckpt,
        hubert_layer,
        quantizer_ckpt,
        is_linear_quantizer=True,
        min_chunk=400,
        max_chunk=100 * 16_000,
    ):
        super().__init__()

        # hubert model
        self.hubert_ckpt = str(hubert_ckpt)
        self.hubert_layer = hubert_layer
        self.hubert_model = None
        self.should_normalize = False
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk

        # quantizer model
        self.quantizer_ckpt = str(quantizer_ckpt)
        self.is_linear_quantizer = is_linear_quantizer
        self.quantizer_model = None

        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))
        self.load_models()

    @torch.no_grad()  # otherwise some non-leaf nodes appear which breaks serialization
    def load_models(self):
        # Load hubert model
        hubert_model, model_cfg, task_cfg = load_hubert_model(self.hubert_ckpt)
        self.hubert_task_cfg = task_cfg
        self.hubert_model_cfg = model_cfg
        self.hubert_model = hubert_model
        self.hubert_model.to(self.device)
        self.hubert_model.eval()
        for parameter in self.hubert_model.parameters():
            parameter.requires_grad_(False)
        self.should_normalize = task_cfg.normalize

        # Load quantizer model
        self.quantizer_model = load_quantizer_model(
            self.quantizer_ckpt, is_linear_quantizer=self.is_linear_quantizer
        )
        self.quantizer_model.to(self.device)
        self.quantizer_model.eval()

    @property
    def device(self):
        return self._float_tensor.device

    @property
    def code_hop_size(self) -> int:
        hop_size = 1
        for dim, kernel, stride in eval(self.hubert_model_cfg.conv_feature_layers):
            hop_size *= stride
        return hop_size  # 320 for 50hz model and 640 for 25hz model

    @property
    def frame_rate(self) -> int:
        return self.expected_sample_rate / self.code_hop_size  # 50 or 25

    @property
    def n_units(self) -> int:
        return self.kmeans_model.K

    @property
    def expected_sample_rate(self) -> int:
        return self.hubert_task_cfg.sample_rate  # 16_000

    def load_audio(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.expected_sample_rate:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=self.expected_sample_rate
            )
        return wav

    @torch.inference_mode()
    def forward(self, x, separate_channels=False, dense=False):
        if isinstance(x, str):
            x = self.load_audio(x)
        i_ndim = x.dim()
        if i_ndim == 2:
            x = x.unsqueeze(0)
        elif i_ndim == 1:
            x = x.view(1, 1, -1)

        # x should expect a shape [B, C, T], where C is number of channels
        assert len(x.shape) == 3
        feats = self.get_dense_features(x)  # [B, T_enc]

        if dense:
            return feats

        tokens = self.quantizer_model(feats)  # [B, T_enc]

        if i_ndim == 3:
            tokens = tokens.view(x.shape[0], 1, -1)
        else:
            tokens = tokens.squeeze(0)

        if not separate_channels:
            return tokens

    @torch.inference_mode()
    def get_dense_features(self, x, separate_channels=False):
        x = x.to(self.device)

        assert separate_channels == False, "Not supported yet"  # TODO: Fix this

        if not separate_channels:
            x = x.mean(1)  # [B, T]

        if self.should_normalize:
            x = torch.cat([nn.functional.layer_norm(item, item.shape) for item in x])

        feat = []
        for start in range(0, x.size(1), self.max_chunk):
            x_chunk = x[:, start : start + self.max_chunk]
            if x_chunk.size(1) < self.min_chunk:
                continue
            feat_chunk, _ = self.hubert_model.extract_features(
                source=x_chunk,
                padding_mask=None,
                mask=False,
                output_layer=self.hubert_layer,
            )
            feat.append(feat_chunk)

        return torch.cat(feat, 1)
