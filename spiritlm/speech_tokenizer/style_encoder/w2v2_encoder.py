# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification


class Wav2Vec2StyleEncoder(Wav2Vec2ForSequenceClassification):
    def __init__(self, config, pool_size: int = 50):
        super().__init__(config)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        self.pool_size = pool_size

        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    @torch.no_grad()
    def forward(self, wavs: Union[torch.Tensor, str]) -> torch.Tensor:
        if isinstance(wavs, str):
            # TODO: resampling if applicable
            wavs = torchaudio.load(wavs)[0].squeeze(0)
        # TODO: handle list of strs
        inputs = self.feature_extractor(
            wavs, sampling_rate=16_000, return_tensors="pt"
        ).input_values
        outputs = self.wav2vec2(inputs.to(self.device))
        hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)
        chunk_size = self.pool_size
        batch_size, sequence_length, hidden_size = hidden_states.shape
        pooled_output = []
        for i in range(0, sequence_length, chunk_size):
            chunk = hidden_states[:, i : i + chunk_size, :]
            pooled_output.append(chunk.mean(dim=1))
        pooled_output = torch.cat(
            pooled_output, dim=1
        )  # Concatenate the chunks along the desired dimension
        pooled_output = pooled_output.view(
            batch_size, -1, hidden_size
        )  # Reshape back to the original shape
        logits = self.classifier(pooled_output)
        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        pred = torch.argmax(lprobs, dim=-1).squeeze(0)
        return pred
