#!/bin/bash

# to run single experiment
# Slurm sbatch options
#SBATCH -a 0-4
## SBATCH --gres=gpu:volta:1
#SBATCH -n 10

# Loading the required module
source /etc/profile
module load anaconda/2020a 

mkdir -p wandb_node2vec_PROTEINS
mkdir -p out_files

# Run the script
node_attr=('True' 'True' 'False' 'False')
node_feat=('True' 'False' 'True' 'False')
python -m node_embedding.runNode2Vec\
--exp_name='proteins'\
--use_node_attr=${node_attr[$SLURM_ARRAY_TASK_ID]} \
--use_node_feat=${node_feat[$SLURM_ARRAY_TASK_ID]} \
&> out_files/${node_attr[$SLURM_ARRAY_TASK_ID]}_${node_feat[$SLURM_ARRAY_TASK_ID]}