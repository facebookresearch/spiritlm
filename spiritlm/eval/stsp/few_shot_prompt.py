# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from enum import Enum, auto
from pathlib import Path

import pandas as pd
import torchaudio
from spiritlm.model.spiritlm_model import Spiritlm

DATA_ROOT = Path(__file__).parents[3] / "data"
FEW_SHOT_MANIFEST_DIR = DATA_ROOT / "manifest/few_shot"
FEW_SHOT_TEMPLATE = "{prompt}{generation}"


class ModalityDirection(Enum):
    T2T = auto()
    S2S = auto()
    S2T = auto()
    T2S = auto()


def build_few_shot_prompt(
    spiritlm_model: Spiritlm,
    modality_direction: ModalityDirection,
    n_shots: int = 3,
) -> str:
    """
    Build the few-shot prompt by simply concatenating a set of examples.

    E.g., a 3-shots T->S prompt would like this:
    "[Text]text1[Speech]speech_tokens1\n[Text]text2[Speech]speech_tokens2\n[Text]text3[Speech]speech_tokens3\n"
    """

    def wav_prompt(wav_path: str, load_first_half: bool) -> str:
        wav_path = DATA_ROOT / wav_path
        wav = torchaudio.load(wav_path)[0].squeeze(0)
        size = wav.size()[0]
        half_size = size // 2
        if load_first_half:
            wav = wav[:half_size]
        else:
            wav = wav[half_size:]
        return spiritlm_model.SPEECH_PROMPT_PREFIX + spiritlm_model.speech_tokenizer(
            wav
        )

    def text_prompt(text: str) -> str:
        return spiritlm_model.TEXT_PROMPT_PREFIX + text

    manifest_path = (
        FEW_SHOT_MANIFEST_DIR / f"{str(modality_direction.name).lower()}.jsonl"
    )
    df = pd.read_json(manifest_path, lines=True)
    assert n_shots <= len(df)

    # ensure a balanced sampels for each sentiment
    nb_samples_per_sentiment = math.ceil(n_shots / 3)
    df = df.groupby("sentiment").sample(n=nb_samples_per_sentiment)

    prompts = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        generation = row["generation"]
        if modality_direction == ModalityDirection.T2T:
            prompt = FEW_SHOT_TEMPLATE.format(
                prompt=text_prompt(prompt),
                generation=text_prompt(generation),
            )
        elif modality_direction == ModalityDirection.T2S:
            prompt = FEW_SHOT_TEMPLATE.format(
                prompt=text_prompt(prompt),
                generation=wav_prompt(generation, load_first_half=False),
            )
        elif modality_direction == ModalityDirection.S2T:
            prompt = FEW_SHOT_TEMPLATE.format(
                prompt=wav_prompt(prompt, load_first_half=True),
                generation=text_prompt(generation),
            )
        elif modality_direction == ModalityDirection.S2S:
            prompt = FEW_SHOT_TEMPLATE.format(
                prompt=wav_prompt(prompt, load_first_half=True),
                generation=wav_prompt(generation, load_first_half=False),
            )
        prompts.append(prompt)
    return "\n".join(prompts) + "\n"
