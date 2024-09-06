# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Usage example:

cd {SPIRITLM ROOT FOLDER}
export PYTHONPATH=.

# Speech to Text
torchrun --nnodes 1 --nproc-per-node 1 spiritlm/eval/stsp/predict_stsp.py --eval_manifest_dir data/stsp_data/records_emov_demo.jsonl --eval --write_pred ./pred_s_t.jsonl --input_output speech_text
# Text to Text
torchrun --nnodes 1 --nproc-per-node 1 spiritlm/eval/stsp/predict_stsp.py --eval_manifest_dir data/stsp_data/records_asr_expresso_demo.jsonl --eval --write_pred ./pred_t_t.jsonl --input_output text_text
# Text to Speech#
torchrun --nnodes 1 --nproc-per-node 1 spiritlm/eval/stsp/predict_stsp.py --eval_manifest_dir data/stsp_data/records_asr_expresso_demo.jsonl --eval --write_pred ./pred._t_s.jsonl --input_output text_speech
# Speech to Speech
torchrun --nnodes 1 --nproc-per-node 1 spiritlm/eval/stsp/predict_stsp.py --eval_manifest_dir data/stsp_data/records_emov_demo.jsonl --eval --write_pred ./pred_s_s.jsonl --input_output speech_speech



# TEXT TO TEXT
"""

import sys

sys.path.append(".")

import argparse
import json
import os

import torch
import torch.distributed as dist
from spiritlm.eval.eval_stsp import eval
from spiritlm.eval.load_data import SpeechData, TextData
from spiritlm.eval.stsp.few_shot_prompt import ModalityDirection, build_few_shot_prompt
from spiritlm.eval.stsp.sentiment_classifiers import (
    get_text_sentiment_prediction,
    load_sentiment_classifier,
)
from spiritlm.eval.stsp.utils import (
    ExpressoEmotionClassifier,
    load_emotion_classifier,
    wav2emotion,
    wav2emotion_and_sentiment,
)
from spiritlm.model.spiritlm_model import (
    ContentType,
    GenerationInput,
    OutputModality,
    Spiritlm,
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, GenerationConfig, set_seed

SPEECH_CLASSIFIER = (
    "/fsx-checkpoints/bokai/speech_gpt/emotion_eval/model/speech_classifier/"
)
TEXT_CLASSIFIER = (
    "/fsx-checkpoints/bokai/speech_gpt/emotion_eval/model/text_classifier/"
)
DATA_ROOT = "data/stsp_data"


def get_eval_classifier(args):
    if args.input_output.endswith("speech"):
        return load_emotion_classifier(SPEECH_CLASSIFIER)
    elif args.input_output.endswith("text"):
        return load_sentiment_classifier(TEXT_CLASSIFIER)
    else:
        raise (Exception(f"{args.input_output} not supported"))


def get_sentiment(
    input_output,
    generation,
    classifer: [AutoModelForSequenceClassification, ExpressoEmotionClassifier],
):
    generation = generation[0].content
    if input_output.endswith("speech"):
        return wav2emotion_and_sentiment(generation, classifer)
    if input_output.endswith("text"):
        return get_text_sentiment_prediction(generation, classifer)


def write_jsonl(dir: str, predictions: dict):
    with open(dir, "w") as f:
        for id, pred in predictions.items():
            record = {"id": id, "pred": pred}
            json_string = json.dumps(record)
            f.write(json_string + "\n")  # Add a newline to separate JSON objects
    print(f"{dir} written")


def run(args, seed: int = 0):
    world_size = int(os.environ["WORLD_SIZE"])
    world_rank = int(os.environ["RANK"])
    print(
        f"Running distributed inference with world_size: {world_size}, world_rank: {world_rank}"
    )
    dist.init_process_group("nccl", rank=world_rank, world_size=world_size)

    set_seed(seed)
    spiritlm_model = Spiritlm(args.model)
    evaluation_classifier = get_eval_classifier(args)

    if args.few_shot > 0:
        raise (NotImplementedError(""))
        prompts = build_few_shot_prompt(
            spiritlm_model=spiritlm_model,
            modality_direction=ModalityDirection.S2S,
            n_shots=3,
        )

    # load
    if args.input_output.startswith("speech"):
        eval_dataset = SpeechData(args.eval_manifest_dir, root_dir=DATA_ROOT)
    elif args.input_output.startswith("text"):
        eval_dataset = TextData(args.eval_manifest_dir, root_dir=DATA_ROOT)

    sampler = DistributedSampler(dataset=eval_dataset)
    loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,  # don't change
        sampler=sampler,
        num_workers=4,
    )
    predictions = {}

    for idx, data in tqdm(
        enumerate(loader),
        desc=f"Predict {args.eval_manifest_dir}",
        total=eval_dataset.__len__() // args.batch_size,
    ):
        out = spiritlm_model.generate(
            output_modality=(
                OutputModality.SPEECH
                if args.input_output.endswith("speech")
                else OutputModality.TEXT
            ),
            interleaved_inputs=[
                GenerationInput(
                    content=(
                        data["wav"][0]
                        if args.input_output.startswith("speech")
                        else data["text"][0]
                    ),  # 0 because of batch size 1
                    content_type=(
                        ContentType.SPEECH
                        if args.input_output.startswith("speech")
                        else ContentType.TEXT
                    ),
                )
            ],
            generation_config=GenerationConfig(
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=200,
                do_sample=True,
            ),
        )
        assert len(out) == 1
        detected_sentiment = get_sentiment(
            args.input_output, out, evaluation_classifier
        )
        predictions[str(data["id"][0])] = detected_sentiment[1]

    if args.write_pred is not None:
        write_jsonl(args.write_pred, predictions)
        # (Pdb) evaluation_classifier.label_names ['angry', 'default', 'happy', 'sad'(Pdb)

    if args.eval:
        eval(
            args.eval_manifest_dir,
            predictions,
            info_data=f"{args.eval_manifest_dir}, input-output {args.input_output}",
            label="sentiment",
        )


def setup_env():
    os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_manifest_dir",  # data/stsp_data/records_emov_demo.jsonl
        type=str,
        help="Path to reference record",
        required=True,
    )

    parser.add_argument(
        "--model",  # data/stsp_data/records_emov_demo.jsonl
        type=str,
        default="spirit-lm-base-7b",
        help="Path to reference record",
        required=False,
    )
    parser.add_argument(
        "--few_shot",  # data/stsp_data/records_emov_demo.jsonl
        type=int,
        default=0,
        help="Path to reference record",
        required=False,
    )
    parser.add_argument(
        "--batch_size",  # data/stsp_data/records_emov_demo.jsonl
        type=int,
        default=1,
        help="",
        required=False,
    )
    parser.add_argument(
        "--input_output",  # data/stsp_data/records_emov_demo.jsonl
        type=str,
        default="speech_speech",
        help="speech_speech speech_text text_speech text_text",
        required=False,
    )
    parser.add_argument(
        "--eval_type",  # data/stsp_data/records_emov_demo.jsonl
        type=str,
        default="emotion",
        required=False,
    )
    parser.add_argument(
        "--write_pred",  # data/stsp_data/records_emov_demo.jsonl
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--eval",  # data/stsp_data/records_emov_demo.jsonl
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    setup_env()
    run(args)
