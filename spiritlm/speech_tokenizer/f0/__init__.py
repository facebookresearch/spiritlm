# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

from pathlib import Path
import os

import torch

from .f0_tokenizer import F0Tokenizer

# Get the base checkpoints directory from environment variable or use the default base path
base_checkpoints_dir = Path(os.getenv("SPIRITLM_CHECKPOINTS_DIR", Path(__file__).parents[3] / "checkpoints"))

# Append 'speech_tokenizer' to the base path
CHECKPOINT_DIR = base_checkpoints_dir / "speech_tokenizer"

CURRENT_DEVICE = (
    torch.device(torch.cuda.current_device())
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def spiritlm_expressive_f0(f0_backbone="fcpe"):
    return F0Tokenizer(
        f0_extractor_method=f0_backbone,
        quantizer_path=CHECKPOINT_DIR / "vqvae_f0_quantizer/model.pt",
        hop_length=80,
        sampling_rate=16000,
        interpolate=True,
        device=CURRENT_DEVICE,
    )
