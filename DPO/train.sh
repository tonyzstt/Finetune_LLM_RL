#!/bin/bash
#SBATCH --account viscam 
#SBATCH --job-name optimize_dpo
#SBATCH --partition=viscam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 
#SBATCH --mem=64G
#SBATCH --nodelist=viscam11
#SBATCH --output=/viscam/u/tonyzst/Research/test/DPO/slurm_outs/%j.out
#SBATCH --error=/viscam/u/tonyzst/Research/test/DPO/slurm_outs/%j.err

cd /viscam/u/tonyzst/Research/test/DPO
mkdir -p /viscam/u/tonyzst/Research/test/DPO/slurm_outs
source ~/.bashrc
conda activate test

export WANDB_API_KEY="847d1d4328b98f89a719f0957806e553c8ca18c6"

echo "Start training..."
python train_16.py
echo "Training finished."