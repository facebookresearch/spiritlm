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
from datasets import load_dataset
import pandas as pd


def pred_to_label(
    sentiment_prediction_scores: List[List[Dict[str, Any]]],
) -> Tuple[str, float]:
    if isinstance(sentiment_prediction_scores[0], list):
        sentiment_prediction_scores = sentiment_prediction_scores[0]
    item_with_max_score = max(
        sentiment_prediction_scores, key=lambda _dict: _dict["score"]
    )
    score = item_with_max_score["score"]
    return score, item_with_max_score["label"].lower()


def get_text_sentiment_prediction(text: str, sentiment_classifier) -> Tuple[str, float]:
    return pred_to_label(sentiment_classifier(text))


def load_sentiment_classifier(model_dir: str):
    classifier = pipeline(
        task="text-classification",
        model=AutoModelForSequenceClassification.from_pretrained(model_dir),
        tokenizer=AutoTokenizer.from_pretrained(
            "j-hartmann/sentiment-roberta-large-english-3-classes"
        ),
        top_k=None,
    )
    return classifier
