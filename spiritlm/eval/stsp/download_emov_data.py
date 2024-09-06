# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tarfile
from multiprocessing.pool import ThreadPool
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parents[3] / "data" / "emov"


def download_and_unzip(url_dir_path):
    url, dir_path = url_dir_path

    tar_filename = url.split("/")[-1]
    tar_path = dir_path / tar_filename

    speaker, emotion = tar_filename.split(".")[0].split("_")
    dst_dir_path = dir_path / speaker / emotion
    dst_dir_path.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(tar_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dst_dir_path)
            os.remove(tar_path)


def download_many(links, save_paths):
    with ThreadPool(16) as pool:
        for _ in pool.imap_unordered(download_and_unzip, zip(links, save_paths)):
            pass


def download_emov(root_dir_path: Path):
    # The links are copied from https://github.com/numediart/EmoV-DB/blob/master/emov_mfa_alignment.py
    download_links = [
        "https://www.openslr.org/resources/115/bea_Amused.tar.gz",
        "https://www.openslr.org/resources/115/bea_Angry.tar.gz",
        "https://www.openslr.org/resources/115/bea_Neutral.tar.gz",
        "https://www.openslr.org/resources/115/jenie_Amused.tar.gz",
        "https://www.openslr.org/resources/115/jenie_Angry.tar.gz",
        "https://www.openslr.org/resources/115/jenie_Neutral.tar.gz",
        "https://www.openslr.org/resources/115/sam_Amused.tar.gz",
        "https://www.openslr.org/resources/115/sam_Angry.tar.gz",
        "https://www.openslr.org/resources/115/sam_Neutral.tar.gz",
        "https://www.openslr.org/resources/115/sam_Sleepy.tar.gz",
    ]
    save_paths = [root_dir_path] * len(download_links)
    download_many(download_links, save_paths)


if __name__ == "__main__":
    print(f"Downloading EmoV dataset to {DATA_DIR}...")
    download_emov(DATA_DIR)
