from collections import defaultdict
from dataclasses import dataclass
from functools import cache
import os
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import torch
import torchaudio
from transformers import (
    pipeline,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


import pandas as pd


EXPRESSO_EMOTION_2_SENTIMENT = {
    "happy": "positive",
    "angry": "negative",
    "sad": "negative",
    "default": "neutral",
}

EMOTION_2_SENTIMENT = {
    "happy": "positive",
    "angry": "negative",
    "sad": "negative",
    "default": "neutral",
    "neutral": "neutral",
    "amused": "positive",
}


@cache
def emotions2new_label_names_and_indices(
    emotions_to_select: Tuple[str],
    label_names: Tuple[str],
) -> Tuple[List[str], List[int]]:
    emotion2index = {e: i for i, e in enumerate(label_names)}
    sorted_indices_emotions = sorted(
        [(emotion2index[emotion], emotion) for emotion in emotions_to_select]
    )
    zipped = list(zip(*sorted_indices_emotions))
    return zipped


def expresso_emotion2_sentiment(emotion: str):
    return EXPRESSO_EMOTION_2_SENTIMENT[emotion]


@dataclass
class ExpressoEmotionClassifier:
    feature_extractor: AutoFeatureExtractor
    model: AutoModelForAudioClassification
    label_names: List[str]


def load_emotion_classifier(checkpoint_path: str) -> ExpressoEmotionClassifier:
    feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint_path)
    model = (
        AutoModelForAudioClassification.from_pretrained(checkpoint_path).cuda().eval()
    )
    label_names = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"Classification model loaded from {checkpoint_path} !")
    return ExpressoEmotionClassifier(feature_extractor, model, label_names)


@torch.inference_mode()
def predict_audio(
    audio,
    expresso_emotion_classifier: ExpressoEmotionClassifier,
    emotions_to_predict: Optional[List[str]] = None,
):
    if isinstance(audio, str):
        speech, _ = torchaudio.load(audio)
        resampler = torchaudio.transforms.Resample(
            expresso_emotion_classifier.feature_extractor.sampling_rate
        )
        speech = resampler(speech).squeeze().numpy()
    else:
        speech = audio

    features = expresso_emotion_classifier.feature_extractor(
        speech,
        sampling_rate=expresso_emotion_classifier.feature_extractor.sampling_rate,
        return_tensors="pt",
    )
    features["input_values"] = features["input_values"].cuda()

    logits = expresso_emotion_classifier.model(**features).logits
    if emotions_to_predict is not None:
        (indices, label_names) = emotions2new_label_names_and_indices(
            tuple(emotions_to_predict), tuple(expresso_emotion_classifier.label_names)
        )
        logits = logits[:, indices]
    else:
        label_names = expresso_emotion_classifier.label_names
    pred_id = torch.argmax(logits, dim=-1)[0].item()

    return label_names[pred_id], logits.detach().cpu().numpy()


def wav2emotion(
    wav,
    expresso_emotion_classifier: ExpressoEmotionClassifier,
    emotions_to_predict: Optional[List[str]] = None,
) -> str:
    label_logits = predict_audio(
        audio=wav,
        expresso_emotion_classifier=expresso_emotion_classifier,
        emotions_to_predict=emotions_to_predict,
    )
    pred_emotion = label_logits[0]
    return pred_emotion


def wav2emotion_and_sentiment(
    wav,
    expresso_emotion_classifier: ExpressoEmotionClassifier,
    emotions_to_predict: Optional[List[str]] = None,
) -> Tuple[str, str]:
    pred_emotion = wav2emotion(wav, expresso_emotion_classifier, emotions_to_predict)
    mapped_sentiment = expresso_emotion2_sentiment(pred_emotion)
    return pred_emotion, mapped_sentiment
