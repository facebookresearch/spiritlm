import os
import re
from io import BytesIO
from typing import List, Optional, Union

import numpy as np
import torch
import torchaudio

EXPECTED_SAMPLING_RATE = 16_000


def find_prompt_last_speech_start_position(prompt: str) -> Optional[int]:
    prev_end = None
    # revert the prompt so we can search from right to left, the speech token patterns are also reverted.
    for match in re.finditer("(\]\d+uH\[)|(\]\d+iP\[)|(\]\d+tS\[)", prompt[::-1]):
        start, end = match.start(), match.end()
        if prev_end is not None and start != prev_end:
            return len(prompt) - prev_end
        prev_end = end
    if prev_end is None:
        # speech token is not found in the prompt
        return None
    return len(prompt) - prev_end


def convert_to_wav_tensor(
    content: Union[str, os.PathLike, torch.Tensor, np.ndarray]
) -> torch.Tensor:
    if isinstance(content, os.PathLike) or isinstance(content, str):
        audio_path = str(content)
        wav, sr = torchaudio.load(audio_path)
        if sr != EXPECTED_SAMPLING_RATE:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=EXPECTED_SAMPLING_RATE
            )
    elif isinstance(content, np.ndarray):
        wav = torch.from_numpy(content)
    elif isinstance(content, bytes):
        wav, sr = torchaudio.load(BytesIO(content))
        if sr != EXPECTED_SAMPLING_RATE:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=EXPECTED_SAMPLING_RATE
            )
    else:
        wav = content

    # TODO: what about stereo ?

    return wav.squeeze()


def does_start_with_speech_token(encoded_string) -> bool:
    if (
        encoded_string is None or len(encoded_string) <= 4
    ):  # shortest speech token is "[Hu1]"
        return False
    if encoded_string[0] != "[":
        return False
    end_pos = 1
    while end_pos < len(encoded_string):
        if encoded_string[end_pos] == "]" and end_pos >= 4:
            if any(encoded_string[1:3].startswith(tok) for tok in ["Hu", "Pi", "St"]):
                return True
            return False
        # longest speech token is "[Huxxxxx]"
        if end_pos >= 10:
            return False
        end_pos += 1
    return False


def does_end_with_speech_token(encoded_string: str) -> bool:
    if (
        encoded_string is None or len(encoded_string) <= 4
    ):  # shortest speech token is "[Hu1]"
        return False
    if encoded_string[-1] != "]":
        return False
    start_pos = len(encoded_string) - 2
    while start_pos >= 0:
        if encoded_string[start_pos] == "[" and start_pos + 3 < len(encoded_string):
            if any(
                encoded_string[start_pos + 1 : start_pos + 3].startswith(tok)
                for tok in ["Hu", "Pi", "St"]
            ):
                return True
            return False
        # longest speech token is "[Huxxxxx]"
        if start_pos < len(encoded_string) - 10:
            return False
        start_pos -= 1
    return False


def get_forbidden_tokens(
    ban_special_tokens: bool = True,
    generate_only_speech: bool = False,
    generate_only_text: bool = False,
    ban_expressivity_tokens: bool = False,
) -> List[int]:
    assert not (
        generate_only_speech and generate_only_text
    ), "Nothing will be generated when generate_only_speech and generate_only_text is all True."
    forbidden_tokens = []
    if ban_special_tokens:
        forbidden_tokens += [
            32000,
            32001,
        ]  # [Text], [Speech]
    if generate_only_speech:
        forbidden_tokens += list(range(32000))
    elif generate_only_text:
        forbidden_tokens += list(range(32002, 32002 + 501))  # hubert tokens
        if ban_expressivity_tokens:
            forbidden_tokens += list(range(32503, 32503 + 64))  # pitch tokens
            forbidden_tokens += list(
                range(32567, 32567 + 100)
            )  # forbidden style tokens
    return forbidden_tokens
