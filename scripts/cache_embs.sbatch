#!/bin/bash
#SBATCH --job-name=cache
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --partition=student

nvidia-smi
nvidia-smi -L

source ~/miniconda3/etc/profile.d/conda.sh
conda activate plugen4faces

set -o xtrace  # autoprint command before running
python3 scripts/evaluate.py cache-embs
