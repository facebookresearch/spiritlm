# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.


import logging
import os
from pathlib import Path

import torch

from .w2v2_encoder import Wav2Vec2StyleEncoder

_logger = logging.getLogger(__name__)

# Get the base checkpoints directory from environment variable or use the default base path
base_checkpoints_dir = Path(os.getenv("SPIRITLM_CHECKPOINTS_DIR", Path(__file__).parents[3] / "checkpoints"))

# Append 'speech_tokenizer' to the base path
CHECKPOINT_DIR = base_checkpoints_dir / "speech_tokenizer"

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
    _logger.info(f"Style encoder loaded from {str(STYLE_ENCODER_CKPT_PATH)}")
    return model
