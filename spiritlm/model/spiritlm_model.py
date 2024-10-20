# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

import logging
import math
import os
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from spiritlm.model.utils import (
    convert_to_wav_tensor,
    does_end_with_speech_token,
    does_start_with_speech_token,
    find_prompt_last_speech_start_position,
    get_forbidden_tokens,
)
from spiritlm.speech_tokenizer import spiritlm_base, spiritlm_expressive
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, set_seed

_logger = logging.getLogger(__name__)


CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints/spiritlm_model"


class ContentType(Enum):
    TEXT = "TEXT"
    SPEECH = "SPEECH"


class OutputModality(Enum):
    TEXT = auto()
    SPEECH = auto()
    ARBITRARY = auto()


@dataclass
class GenerationInput:
    content: Union[str, os.PathLike, torch.Tensor, np.ndarray]
    content_type: ContentType

    @classmethod
    def from_tuple(cls, tup):
        content_type, content = tup
        content_type = content_type.upper()
        assert content_type in [
            "SPEECH",
            "TEXT",
        ], f"expects content_type to be one of ['SPEECH', 'TEXT'], found '{content_type}'"
        if content_type == "TEXT":
            content_type = ContentType.TEXT
        elif content_type == "SPEECH":
            content_type = ContentType.SPEECH
        return cls(content=content, content_type=content_type)


@dataclass
class GenerationOuput:
    content: Union[str, np.ndarray]
    content_type: ContentType


InterleavedInputs = List[GenerationInput]
InterleavedOutputs = List[GenerationOuput]


class SpiritlmVariants(Enum):
    BASE_7B = "spirit-lm-base-7b"
    EXPRESSIVIE_7B = "spirit-lm-expressive-7b"

    @classmethod
    def values_as_list(cls):
        return [e.value for e in cls]


def _ensure_model_name(name: str):
    if Path(name).exists():
        name = Path(name).stem
    expected_names = SpiritlmVariants.values_as_list()
    assert (
        name in SpiritlmVariants.values_as_list()
    ), f"Unknown model name, expected one of {expected_names}"


def _set_device_and_return():
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device(local_rank)


def _convert_str_output_modality(output_modality):
    """Convert from string to an instance of OutputModality"""
    output_modality_str_map = {
        "TEXT": OutputModality.TEXT,
        "SPEECH": OutputModality.SPEECH,
        "ARBITRARY": OutputModality.ARBITRARY,
    }
    if isinstance(output_modality, str):
        output_modality = output_modality.upper()
        assert (
            output_modality in output_modality_str_map
        ), f"invalid string output_modality (found '{output_modality}', but expects one of {list(output_modality_str_map)})"
        output_modality = output_modality_str_map[output_modality]
    assert isinstance(output_modality, OutputModality)
    return output_modality


def _get_generation_inputs(interleaved_inputs):
    """Convert from a list of tuple (content_type, content) to a list of GenrationInput"""
    for i, item in enumerate(interleaved_inputs):
        assert isinstance(item, tuple) or isinstance(item, GenerationInput), (
            "Each element of interleaved_inputs is expected to be either an instance of GenerationInput "
            "or a tuple of (content_modality, content)"
        )
        if isinstance(item, tuple):
            interleaved_inputs[i] = GenerationInput.from_tuple(interleaved_inputs[i])
    return interleaved_inputs


def _overwrite_generation_config(generation_config, kwargs):
    """Overwrite generation_config from the kwargs"""
    if generation_config is None:
        generation_config = GenerationConfig()
    assert isinstance(generation_config, GenerationConfig)
    gen_diff_dict = generation_config.to_diff_dict()
    for attr_name, attr_value in kwargs.items():
        assert hasattr(
            generation_config, attr_name
        ), f"attribute '{attr_name}' not found in transformers.GenerationConfig"
        if attr_name in gen_diff_dict and attr_value != gen_diff_dict[attr_name]:
            _logger.warning(
                f"Overwrite generation_config's {attr_name} to {attr_value}"
            )
        setattr(generation_config, attr_name, attr_value)
    return generation_config


class Spiritlm:
    TEXT_PROMPT_PREFIX = "[Text]"
    SPEECH_PROMPT_PREFIX = "[Speech]"

    def __init__(self, name: str, **speech_tokenizer_kwargs):
        if Path(name).exists():
            path = name
        else:
            path = CHECKPOINT_DIR / name
        _ensure_model_name(name)
        self.device = _set_device_and_return()
        _logger.info(f"Loading SPIRIT-LM model from the path {path}...")
        self.model = LlamaForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16
        ).to(self.device)
        _logger.info(f"SPIRIT-LM model is loaded.")
        self.tokenizer = LlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path=path,
            add_bos_token=True,
            add_eos_token=False,
        )
        _logger.info("Loading SPIRIT-LM speech tokenizers ...")
        if name == SpiritlmVariants.BASE_7B.value:
            self.speech_tokenizer = spiritlm_base(**speech_tokenizer_kwargs)
            self.is_expressive_model = False
        elif name == SpiritlmVariants.EXPRESSIVIE_7B.value:
            self.speech_tokenizer = spiritlm_expressive(**speech_tokenizer_kwargs)
            self.is_expressive_model = True
        _logger.info("SPIRIT-LM speech tokenizers are loaded.")

    def _build_prompt(
        self,
        generation_inputs: List[GenerationInput],
        output_modality: OutputModality,
    ) -> str:
        """
        Build the prompt according the input content and the output modality.
        """
        if not isinstance(output_modality, OutputModality):
            raise ValueError(f"Unknown output_modality: {output_modality}")
        prompts = []
        prev_modality = None
        for gen_input in generation_inputs:
            if gen_input.content_type.value == ContentType.SPEECH.value:
                gen_input.content = convert_to_wav_tensor(gen_input.content)
                if prev_modality != "a":
                    prompts.append(Spiritlm.SPEECH_PROMPT_PREFIX)
                prompts.append(self.speech_tokenizer(gen_input.content))
                prev_modality = "s"  # speech
            elif gen_input.content_type.value == ContentType.TEXT.value:
                if prev_modality != "t":
                    prompts.append(Spiritlm.TEXT_PROMPT_PREFIX)
                prompts.append(gen_input.content)
                prev_modality = "t"  # text
            else:
                raise ValueError(
                    f"Unknown content type: {gen_input.content_type.value}"
                )
        if output_modality == OutputModality.TEXT:
            if prev_modality != "t":
                prompts.append(Spiritlm.TEXT_PROMPT_PREFIX)
        elif output_modality == OutputModality.SPEECH:
            if prev_modality != "s":
                prompts.append(Spiritlm.SPEECH_PROMPT_PREFIX)
        return "".join(prompts)

    @cache
    def _build_forbidden_tokens(
        self,
        output_modality: OutputModality,
    ) -> List[int]:
        """
        Build a set of token ids that we don't want to generate according the modality direction.

        For instance, when the modality direction is speech to text (S2T), i.e., we continue
        generating text given a speech prompt, we want that the output contains only the text tokens.
        """
        if output_modality == OutputModality.TEXT:
            forbidden_tokens = get_forbidden_tokens(
                ban_special_tokens=True,
                generate_only_text=True,
                ban_expressivity_tokens=True if self.is_expressive_model else False,
            )
        elif output_modality == OutputModality.SPEECH:
            forbidden_tokens = get_forbidden_tokens(
                ban_special_tokens=True,
                generate_only_speech=True,
            )
        elif output_modality == OutputModality.ARBITRARY:
            forbidden_tokens = []
        else:
            raise ValueError(f"Unknown output_modality: {output_modality}")
        return forbidden_tokens

    def _parse_speech_and_text(
        self,
        generated_content: str,
    ):
        # TODO: clean this function, it is too long!
        splits = []
        i = 0
        last_pos = len(generated_content)
        char_and_types = []
        is_speech_token = False
        is_text_token = False
        text_prefix_length = len(Spiritlm.TEXT_PROMPT_PREFIX)
        speech_prefix_length = len(Spiritlm.SPEECH_PROMPT_PREFIX)
        while i < last_pos:
            ch = generated_content[i]
            j = i
            if ch == "[":
                if (
                    j + text_prefix_length - 1 < last_pos
                    and generated_content[j : j + text_prefix_length]
                    == Spiritlm.TEXT_PROMPT_PREFIX
                ):  # text prefix token
                    j += text_prefix_length  # skip "[Text]
                elif (
                    j + speech_prefix_length - 1 < last_pos
                    and generated_content[j : j + speech_prefix_length]
                    == Spiritlm.SPEECH_PROMPT_PREFIX
                ):  # speech prefix token
                    j += speech_prefix_length  # skip "[Speech]"
                elif j + 2 < last_pos and generated_content[j + 1 : j + 3] in (
                    "Hu",
                    "Pi",
                    "St",
                ):
                    j += 3  # skip "["" and Hu/Pi/St
                    while j < last_pos and generated_content[j] != "]":
                        j += 1
                    j += 1  # skip "]"
                    is_speech_token = True
                else:  # other texts starting with "[" e.g., "[abc"
                    is_text_token = True
                    j += 1
            else:
                is_text_token = True
                while j < last_pos and generated_content[j] != "[":
                    j += 1

            cur_content = generated_content[i:j]
            if is_speech_token:
                if len(char_and_types) and char_and_types[-1][1] == "t":
                    splits.append(
                        (
                            "".join(
                                (
                                    content_and_type[0]
                                    for content_and_type in char_and_types
                                )
                            ),
                            "t",
                        )
                    )
                    char_and_types = []
                char_and_types.append((cur_content, "s"))  # speech
            elif is_text_token:
                if len(char_and_types) and char_and_types[-1][1] == "s":
                    splits.append(
                        (
                            "".join(
                                (
                                    content_and_type[0]
                                    for content_and_type in char_and_types
                                )
                            ),
                            "s",
                        )
                    )
                    char_and_types = []
                char_and_types.append((cur_content, "t"))  # text
            is_speech_token, is_text_token = False, False
            i = j
        if len(char_and_types):
            if char_and_types[-1][1] == "t":
                splits.append(
                    (
                        "".join(
                            (content_and_type[0] for content_and_type in char_and_types)
                        ),
                        "t",
                    )
                )
            else:
                splits.append(
                    (
                        "".join(
                            (content_and_type[0] for content_and_type in char_and_types)
                        ),
                        "s",
                    )
                )
        return splits

    def _decode_from_generated_output(
        self,
        output_modality: OutputModality,
        generated_content: str,
        prompt: str,
        speaker_id: int = 2,
    ) -> InterleavedOutputs:
        """
        Decode the generated tokens according the modality direction.

        If the output is text, we return what it is.
        If the output is speech, we decode speech tokens by the speech tokenizer.
        If the output is arbitrary, we decode the generated content according to the its modality.
        """

        def _decode(
            modality: OutputModality,
            gen: str,
        ) -> InterleavedOutputs:
            if modality == OutputModality.TEXT:
                return [
                    GenerationOuput(
                        content=gen,
                        content_type=ContentType.TEXT,
                    )
                ]
            elif modality == OutputModality.SPEECH:
                return [
                    GenerationOuput(
                        content=self.speech_tokenizer.decode(
                            gen, speaker_id=speaker_id
                        ),
                        content_type=ContentType.SPEECH,
                    )
                ]
            elif modality == OutputModality.ARBITRARY:
                decoded_chunks = []
                for i, (chunk_content, chunk_modality) in enumerate(
                    self._parse_speech_and_text(gen)
                ):
                    if chunk_modality == "s":
                        # TODO: the way of finding Hubert token could be false positive
                        nb_content_hubert_tokens = len(chunk_content.split("[Hu"))
                        decoded = _decode(
                            modality=OutputModality.SPEECH,
                            gen=chunk_content,
                        )[0]
                        if i == 0 and is_last_content_speech:
                            # edge case when the prompt ends with speech and the generation starts with speech
                            nb_prompt_hubert_tokens = (
                                len(prompt[last_speech_start_pos:].split("[Hu")) - 1
                            )  # minus the one in prefix
                            if nb_content_hubert_tokens - nb_prompt_hubert_tokens < 25:
                                # continued speech from the prompt is too short
                                continue
                            # we drop the prompt part from the generation
                            prompt_ratio = (
                                nb_prompt_hubert_tokens / nb_content_hubert_tokens
                            )
                            decoded.content = decoded.content[
                                math.ceil(decoded.content.size * prompt_ratio) :
                            ]
                        elif i > 0 and nb_content_hubert_tokens < 25:
                            # new speech in generation is too short
                            continue
                    else:
                        decoded = _decode(
                            modality=OutputModality.TEXT,
                            gen=chunk_content,
                        )[0]
                    decoded_chunks.append(decoded)
                return decoded_chunks
            else:
                raise ValueError(f"Unknown output_modality: {output_modality}")

        generated_new_content = generated_content[len(prompt) :].strip()
        is_last_content_speech, last_speech_start_pos = False, 0
        if (
            output_modality == OutputModality.ARBITRARY
            and does_end_with_speech_token(prompt)
            and does_start_with_speech_token(generated_new_content)
        ):
            is_last_content_speech = True
            last_speech_start_pos = find_prompt_last_speech_start_position(prompt)
            # If the prompt ends with speech, we decode both the prompt and the generation
            # because we probably don't have pitch and style tokens in the generation.
            generated_new_content = generated_content[last_speech_start_pos:]
        return _decode(output_modality, generated_new_content)

    def generate(
        self,
        interleaved_inputs: Optional[List[Union[GenerationInput, tuple]]] = None,
        prompt: Optional[str] = None,
        output_modality: Union[OutputModality, str] = OutputModality.ARBITRARY,
        generation_config: Optional[GenerationConfig] = None,
        force_tokens_to_output_modality: bool = True,
        speaker_id: int = 2,
        return_prompt: bool = False,
        seed: Optional[int] = None,
        **kwargs,  # GenerationConfig args can be passing here
    ) -> Union[InterleavedOutputs, Tuple[InterleavedOutputs, str]]:
        """
        Speech/text generation given speech/text prompt.

        Parameters:
            interleaved_inputs (List of `GenerationInput` or list of tuples):
                List of speech/text inputs.
                Each element can be an instance of `GenerationInput` or a tuple of (content_type, content)
                Text content is string; Speech content is either audio path, audio tensor, or nummpy array.
                The prompt will be built by interleaving them in order.
            prompt (str):
                The prompt in encoded tokens string,
                e.g., "[Speech][Hu99][Hu38]...", "[Text]whatever text" or mix of speech & text.
            output_modality (str or `OutputModality`):
                'TEXT' or OutputModality.TEXT: generate text
                'SPEECH' or OutputModality.SPEECH: generate speech
                'ARBITRARY' or OutputModality.ARBITRARY: generate arbitrary modality output (default)
            generation_config (`GenerationConfig`):
                Generation configuration used by Huggingface `generate` function.
            force_tokens_to_output_modality (bool):
                Whether to force generating tokens to the output modality that you specify in `output_modality`.
                For instance, if the `output_modality` is TEXT and force_tokens_to_output_modality is True,
                we force the model to generate only the text tokens.
            speaker_id (int):
                Speaker id, 0, 1, 2 or 3.
            return_prompt (bool):
                Whether to return the constructed prompt (could be used for debug).
            **kwargs:
                Directly passing arguments from transformers.GenerationConfig (e.g. temperature, max_new_tokens, do_sample).
                See: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
        """

        if seed is not None:
            _logger.info(f"Set seed to {seed}")
            set_seed(seed)

        # Set the output modality
        output_modality = _convert_str_output_modality(output_modality)

        # Get the input prompt
        assert not (
            interleaved_inputs is None and prompt is None
        ), "interleaved_inputs and prompt can not both be None"
        if (
            prompt is not None
            and interleaved_inputs is not None
            and len(interleaved_inputs) > 0
        ):
            _logger.warning(
                "When prompt is specified, interleaved_inputs will not be used."
            )
        if prompt is None:
            if not isinstance(interleaved_inputs, list):
                interleaved_inputs = [interleaved_inputs]
            interleaved_inputs = _get_generation_inputs(interleaved_inputs)
            prompt = self._build_prompt(
                interleaved_inputs,
                output_modality,
            )

        # Get input tensor
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Get generation config from kwargs
        generation_config = _overwrite_generation_config(generation_config, kwargs)

        # Get forbidden token ids
        if (
            force_tokens_to_output_modality
            and output_modality != OutputModality.ARBITRARY
        ):
            forbidden_token_ids = [
                [tok_id] for tok_id in self._build_forbidden_tokens(output_modality)
            ]
        else:
            forbidden_token_ids = None

        # Perform the generation
        generate_ids = self.model.generate(
            **inputs,
            generation_config=generation_config,
            bad_words_ids=forbidden_token_ids,
            pad_token_id=-1,
        )

        # Decode the output
        gen = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        try:
            decoded_output = self._decode_from_generated_output(
                output_modality=output_modality,
                generated_content=gen,
                prompt=prompt,
                speaker_id=speaker_id,
            )
        except Exception as e:
            _logger.error(f"Fail to decode the content: {gen[len(prompt) :].strip()}")
            raise e

        if return_prompt:
            return decoded_output, prompt
        else:
            return decoded_output


if __name__ == "__main__":
    spirit_lm = Spiritlm("spirit-lm-expressive-7b")
    # run several time to test speech text interleaved outputs
    wav = torchaudio.load("examples/audio/7143-88743-0029.flac")[0].squeeze()
    for i in range(5):
        outs = spirit_lm.generate(
            output_modality=OutputModality.ARBITRARY,
            interleaved_inputs=[
                GenerationInput(
                    content=wav,
                    content_type=ContentType.SPEECH,
                )
            ],
            generation_config=GenerationConfig(
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=200,
                do_sample=True,
            ),
        )
        print("-" * 100)
        print(i)
        print("-" * 100)
        print(outs)
