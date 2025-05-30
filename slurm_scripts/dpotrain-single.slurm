#!/bin/bash

#SBATCH --job-name=dpoTrainLlama8b
#SBATCH -D .
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18GB
#SBATCH --gres=gpu:1
#SBATCH --output=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/dpoTrainLlama8b-single2.txt
#SBATCH --error=/sorgin1/users/jbarrutia006/viper/slurm_scripts/outputs/dpoTrainLlama8b-single2.err
#SBATCH --mail-type=all
#SBATCH --mail-user=jbarrutia006@ikasle.ehu.eus


source /sorgin1/users/jbarrutia006/venvs/viper_tximista/bin/activate

srun python scripts/dpotrain.py \
    --project_name "viperDPO" \
    --run_name "Codellama single" \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --train_dataset "/sorgin1/users/jbarrutia006/viper/PrefDatasets/dpo_dataset_single_train.arrow" \
    --dev_dataset "/sorgin1/users/jbarrutia006/viper/PrefDatasets/dpo_dataset_single_dev.arrow" \
    --output_dir "./dpo_trained_models" \
    --batch_size 4 \
    --gradient_accumulation 4 \
    --learning_rate 5e-5 \
    --beta 0.1 \
    --weight_decay 0.01 \
    --epochs 6 \
    --max_steps -1 \
    --logging_steps 40 \
    --eval_steps 100 \
    --save_steps 100 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --device "cuda"

