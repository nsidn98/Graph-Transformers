#!/bin/bash

# to run single experiment
# Slurm sbatch options
#SBATCH -a 0-4
#SBATCH --gres=gpu:volta:1
#SBATCH -n 10

# Loading the required module
source /etc/profile
module load anaconda/2020a 

mkdir -p wandb_NT
mkdir -p out_files
# Run the script
# script to iterate through different hyperparameters
datasets=('TU_ENZYMES' 'TU_MUTAG' 'TU_AIDS' 'TU_PROTEINS' 'TU_IMDB')
python -m train_scripts.runNaiveTransformer.py --dataset_name=${datasets[$SLURM_ARRAY_TASK_ID]}  --exp_name=${datasets[$SLURM_ARRAY_TASK_ID]} --batch_size=128 &> out_files/out_${nodes[$SLURM_ARRAY_TASK_ID]}_${Ks[$SLURM_ARRAY_TASK_ID]}_CIFAR
