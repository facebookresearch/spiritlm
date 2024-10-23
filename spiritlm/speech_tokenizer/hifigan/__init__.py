# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

from pathlib import Path
import os

import torch

from .hifigan_vocoder import HifiGANVocoder

# Get the base checkpoints directory from environment variable or use the default base path
base_checkpoints_dir = Path(os.getenv("SPIRITLM_CHECKPOINTS_DIR", Path(__file__).parents[3] / "checkpoints"))

# Append 'speech_tokenizer' to the base path
CHECKPOINT_DIR = base_checkpoints_dir / "speech_tokenizer"

CURRENT_DEVICE = (
    torch.device(torch.cuda.current_device())
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def spiritlm_base_hifigan(
    default_speaker=2,
    default_style=8,  # conv-default
):
    return HifiGANVocoder(
        CHECKPOINT_DIR / "hifigan_spiritlm_base/generator.pt",
        default_speaker=default_speaker,
        default_style=default_style,
    ).to(CURRENT_DEVICE)


def spiritlm_expressive_hifigan_w2v2(default_speaker=2):
    return HifiGANVocoder(
        CHECKPOINT_DIR / "hifigan_spiritlm_expressive_w2v2/generator.pt",
        default_speaker=default_speaker,
    ).to(CURRENT_DEVICE)
