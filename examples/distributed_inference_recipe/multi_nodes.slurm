#!/bin/bash

#SBATCH --job-name=spiritlm
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --nodes=2
#SBATCH --cpus-per-task=12
#SBATCH --output=./logs/%j.stdout
#SBATCH --error=./logs/%j.stderr
#SBATCH --time=01:00:00

set -e

srun bash -c 'torchrun --nnodes $SLURM_JOB_NUM_NODES --nproc-per-node $SLURM_GPUS_ON_NODE \
--node-rank $SLURM_PROCID \
--master-addr $(scontrol show hostnames $SLURM_NODELIST | head -n1) \
--master-port 12345 \
examples/distributed_inference_recipe/run_dist.py'
