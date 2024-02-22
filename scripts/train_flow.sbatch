#!/bin/bash
#SBATCH --job-name=realnvp
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=rtx3080

nvidia-smi
nvidia-smi -L

source ~/miniconda3/etc/profile.d/conda.sh
conda activate plugen4faces
set -o xtrace  # autoprint command before running
python3 scripts/train.py --epochs 1500 --sigma 0.7 --decay 0.9995 --num_layers 10 --batch_size 512
