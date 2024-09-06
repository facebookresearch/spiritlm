# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch

from .w2v2_encoder import Wav2Vec2StyleEncoder

CHECKPOINT_DIR = Path(__file__).parents[3] / "checkpoints/speech_tokenizer"

CURRENT_DEVICE = (
    torch.device(torch.cuda.current_device())
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def spiritlm_expressive_style_encoder_w2v2() -> Wav2Vec2StyleEncoder:
    STYLE_ENCODER_CKPT_PATH = CHECKPOINT_DIR / "style_encoder_w2v2"
    model = Wav2Vec2StyleEncoder.from_pretrained(
        pretrained_model_name_or_path=STYLE_ENCODER_CKPT_PATH
    ).to(CURRENT_DEVICE)
    print(f"Style encoder loaded from {str(STYLE_ENCODER_CKPT_PATH)}")
    return model
