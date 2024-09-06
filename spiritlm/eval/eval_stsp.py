# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import sys

import pandas as pd

sys.path.append(".")

from spiritlm.eval.stsp.utils import EMOTION_2_SENTIMENT


def load_pred(predictions):
    ret = {}
    with open(predictions) as f:
        for line in f:
            pred = json.loads(line)
            ret[pred["id"]] = pred["pred"]

    assert sum(1 for _ in open(predictions)) == len(ret)

    return ret


def eval(gold_records: str, predictions: [str, dict], info_data="", label="emotion"):
    n_gold_records = sum(1 for _ in open(gold_records))
    n_lines_pred = (
        sum(1 for _ in open(predictions))
        if isinstance(predictions, str)
        else len(predictions)
    )
    assert (
        n_gold_records == n_lines_pred
    ), f"Mismatch between prediction ({n_lines_pred} samples in {predictions}) and reference ({n_gold_records} in {gold_records})"

    pred_dic = load_pred(predictions) if isinstance(predictions, str) else predictions

    scores = []

    with open(gold_records) as gold:
        for line in gold:
            ref = json.loads(line)
            try:
                if label in ref:
                    scores.append(pred_dic[str(ref["id"])] == ref[label])
                else:
                    assert label == "sentiment" and "emotion" in ref, ref
                    sentiment = EMOTION_2_SENTIMENT[ref["emotion"]]
                    scores.append(pred_dic[str(ref["id"])] == sentiment)
            except Exception as e:
                print(
                    f"ERROR in matching the predicted labels with the gold ones: {e}: ref['id']  do not match any key in {pred_dic}', {ref['id']}: "
                )
    # TODO: add other metrics if needed : F1 per class, etc.
    report = pd.DataFrame({"Correct": scores})
    if isinstance(predictions, str):
        info_data += f"from {predictions}"
    print(
        f"Accuracy: {(report['Correct']==1).sum()/len(report)*100:0.2f}% for predictions {info_data}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ref_file",
        type=str,
        help="Path to reference record",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        help="Path to prediction: should be jsonl with each entry {'pred': , 'id': }",
    )
    args = parser.parse_args()

    eval(args.ref_file, args.pred_file)
