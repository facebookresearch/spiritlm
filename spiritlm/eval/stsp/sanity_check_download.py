# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
from pathlib import Path


def check_emov(root_emov_records: str, splits: list[str]):
    for split in splits:
        records_checked = 0
        with open(Path(root_emov_records) / f"records_emov_{split}.jsonl") as f:
            for record in f:
                record = json.loads(record)
                assert (
                    root_emov_records / record["wav_path"]
                ).is_file(), f"Record {record['wav_path']} not found in {root_emov_records/record['wav_path']} and listed in {root_emov_records/f'records_emov_{split}.jsonl'}"
                records_checked += 1
        print(f"{records_checked} records checked for emov {split} split")


if __name__ == "__main__":
    src_data = Path("/data/home/benjaminmuller/project/spiritlm/data/stsp_data")
    check_emov(src_data, ["train", "dev", "test"])
