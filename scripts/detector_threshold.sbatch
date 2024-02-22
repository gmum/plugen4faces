#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --partition=rtx3080

nvidia-smi
nvidia-smi -L

source ~/miniconda3/etc/profile.d/conda.sh
conda activate plugen-videos3
export CUDA_HOME=$CONDA_PREFIX
export LIBRARY_PATH=$LIBRARY_PATH:$CONDA_PREFIX/lib

set -o xtrace  # autoprint command before running

#    -e beard 3 0.3 \
#    -e beard -3 0.09 \

#python3 evaluate.py --save_results 1 detector-threshold \
#    -e gender 2 0.8 \
#    -e gender -2 0.2 \
#    -e glasses 2 0.6 \
#    -e glasses -2 0.2 \
#    -e bald 2 0.4 \
#    -e bald -2 0.5 \
#    --sf_subset 0 \
#    --detector_path /shared/results/bwojcik/public/resnet18_sd_11022023_shrnk_std.pt \
#    --detector_classes 9 \
#    --num_steps 20 \
#    --search_batch_size 10 \
#    --save_video 1 \
#    --newmodel_path '' \
#    --plugen_path '' \
#    --styleflow_path /shared/results/z1134376/plugen/StyleGAN/saved/original/feat8_normalized_frames/model_e10.pch \
#    --cache_path evaluation_results/evaluate_cache_embs_2023-02-21_17_10_52.pkl # slurm-36813


#    -e age 2 50.0 \
#    -e age -2 18.0 \
#    -e smile 2 0.8 \
#    -e smile -2 0.3 \
#    -e pitch 2 4.0 \
#    -e pitch -2 -4.0 \
#    -e yaw 2 8.0 \
#    -e yaw -2 -8.0 \


#    -e age 2 50.0 \
#    -e age -2 18.0 \
#    -e smile 2 0.8 \

    #-e yaw 2 8.0 \

ROOT=~/plugen-finetune
#PLUGEN=/shared/results/z1156545/plugen/StyleGAN/saved_bothdatasets_bs512/feat8_valuescontinuous_decay0.999_sigma0.4/model_e1000.pch
PLUGEN=~/plugen-finetune/mlruns/0/a3b7b85c2ef14165a3a72599a06afbf1/artifacts/plugen_finetune_epochs4_model_e1000.pt
CACHE_PATH=~/plugen-videos/evaluation_results/evaluate_cache_embs_2023-02-21_17_10_52.pkl 

    # -e bald 2 0.4 \
    # -e bald -2 0.5 \
    # -e pitch 2 4.0 \
    # -e pitch -2 -4.0 \
    # -e yaw 2 8.0 \
    # -e yaw -2 -8.0 \
    # -e beard 3 0.3 \
    # -e beard -3 0.09 \



python3 scripts/evaluate.py --save_results 1 detector-threshold \
    -e gender 2 0.8 \
    -e gender -2 0.2 \
    -e glasses 2 0.6 \
    -e glasses -2 0.2 \
    -e age 2 50.0 \
    -e age -2 18.0 \
    -e smile 2 0.8 \
    -e smile -2 0.3 \
    --detector_path $ROOT/pretrained_models/resnet18_sd_11022023_shrnk_std.pt \
    --detector_classes 9 \
    --num_steps 20 \
    --search_batch_size 10 \
    --save_video 1 \
    --stylegan_path $ROOT/pretrained_models/stylegan2-ffhq-config-f.pt \
    --plugen4faces_path '' \
    --styleflow_path '' \
    --plugen_path $PLUGEN \
    --cache_path $CACHE_PATH # slurm-36813



# --styleflow_path /shared/results/z1134376/plugen/StyleGAN/saved/original/feat8_normalized_frames/model_e10.pch \
#    --sf_subset 0 \
