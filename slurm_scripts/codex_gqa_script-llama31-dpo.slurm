#!/bin/bash

#SBATCH --job-name=codigo_GQA_dpo
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/codex_llama318b-dpo.txt
#SBATCH --error=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/codex_llama318b-dpo.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus


source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

export LOAD_MODELS=0
export DATASET='gqa'
export EXEC_MODE='codex'
export CODEX_MODEL='llama31Q'
export PARTITION='train'
export BATCHSIZE=32
export TEMP=0
export NUMINST=
export CHECKPOINT='03-15_21-54-00/checkpoint-1050'

srun python src/main_project_slurm.py
