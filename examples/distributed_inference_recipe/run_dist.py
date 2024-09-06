# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Usage example:

cd {SPIRITLM ROOT FOLDER}
export PYTHONPATH=.

Single node, multi-gpus:
(Assume that your machine has 8 GPUs)
    torchrun --nnodes 1 --nproc-per-node 8 examples/distributed_inference_recipe/run_dist.py

Multi-nodes, multi-gpus:
(2 nodes, 8 GPUs for eahc node, via sbatch)
    mkdir -p logs
    sbatch examples/distributed_inference_recipe/multi_nodes.slurm
"""

import os
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torchaudio
from transformers import GenerationConfig, set_seed

from spiritlm.model.spiritlm_model import (
    ContentType,
    GenerationInput,
    OutputModality,
    Spiritlm,
)


def run(seed: int = 0):
    world_size = int(os.environ["WORLD_SIZE"])
    world_rank = int(os.environ["RANK"])
    print(
        f"Running distributed inference with world_size: {world_size}, world_rank: {world_rank}"
    )
    dist.init_process_group("nccl", rank=world_rank, world_size=world_size)

    set_seed(seed)

    wav = torchaudio.load("examples/audio/7143-88743-0029.flac")[0].squeeze()

    # fake repeated dataset
    dataset = TensorDataset(wav.repeat(32, 1))

    sampler = DistributedSampler(dataset=dataset)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,  # don't change
        sampler=sampler,
        num_workers=4,
    )

    spirit_lm = Spiritlm("spirit-lm-expressive-7b")

    for _, data in enumerate(loader):
        outs = spirit_lm.generate(
            output_modality=OutputModality.ARBITRARY,
            interleaved_inputs=[
                GenerationInput(
                    content=data[0],  # 0 because of batch size 1
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
        print(f"outs: {outs}")


def setup_env():
    os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == "__main__":
    setup_env()
    run()
