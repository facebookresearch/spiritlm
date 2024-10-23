# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

from pathlib import Path
import os

import torch

from .hubert_tokenizer import HubertTokenizer

# Get the base checkpoints directory from environment variable or use the default base path
base_checkpoints_dir = Path(os.getenv("SPIRITLM_CHECKPOINTS_DIR", Path(__file__).parents[3] / "checkpoints"))

# Append 'speech_tokenizer' to the base path
CHECKPOINT_DIR = base_checkpoints_dir / "speech_tokenizer"

CURRENT_DEVICE = (
    torch.device(torch.cuda.current_device())
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def spiritlm_hubert():
    return HubertTokenizer(
        hubert_ckpt=CHECKPOINT_DIR / "hubert_25hz/mhubert_base_25hz.pt",
        hubert_layer=11,
        quantizer_ckpt=CHECKPOINT_DIR / "hubert_25hz/L11_quantizer_500.pt",
        is_linear_quantizer=True,
    ).to(CURRENT_DEVICE)
