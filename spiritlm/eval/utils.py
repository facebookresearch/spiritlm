# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchaudio
from spiritlm.model.spiritlm_model import Spiritlm


def wav_prompt(spiritlm_model: Spiritlm, wav_path: str) -> str:
    wav = torchaudio.load(wav_path)[0].squeeze(0)
    return spiritlm_model.SPEECH_PROMPT_PREFIX + spiritlm_model.speech_tokenizer(wav)


def text_prompt(spiritlm_model: Spiritlm, text: str) -> str:
    return spiritlm_model.TEXT_PROMPT_PREFIX + text
