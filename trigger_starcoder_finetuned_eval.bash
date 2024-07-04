#!/bin/bash
#SBATCH --mem=200g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64     # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4-interactive # <- one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account='bbvz-delta-gpu'
#SBATCH --job-name="evaluation_of_finetuned_code"
#SBATCH --time=1:00:00
### GPU options ###
#SBATCH --gpus-per-node=4
# SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
# module load python anaconda3_gpu  # ... or any appropriate modules
# module list  # job documentation and metadata
# echo "job is starting on `hostname`"

source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate scoder_2

# Set the HF_HOME environment variable
export HF_HOME=/scratch/bbvz/choprahetarth
export WANDB_DIR=/scratch/bbvz/choprahetarth  # replace with your desired path
export WANDB_API_KEY=e1b18fcb1054536d8c6958c02a175ddff40f4914
export HF_API_KEY=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
export HF_TOKEN=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

# watch -n 0.5 nvidia-smi >> /u/choprahetarth/all_files/starcoder/gpu_usage.log
echo "Starting the evaluation process..."

# srun --account=bbvz-delta-gpu \
# python -m torch.distributed.run --nproc_per_node=4 /u/choprahetarth/all_files/starcoder/external_evals/starcoder_finetuned_evaluation.py \
# --base_model_name_or_path='bigcode/starcoderbase-7b' \
# --peft_model_path='//projects/bbvz/choprahetarth/new_experiments/experiment_2/final_checkpoint_starcoderbase-7b_lr_0.0001_bs_16_ms_214' \
# --save='/scratch/bbvz/choprahetarth/experiment_2/fused_model' \
# --batch_size=1

# srun --account=bbvz-delta-gpu \
# python -m torch.distributed.run --nproc_per_node=4 /u/choprahetarth/all_files/starcoder/external_evals/starcoder_finetuned_evaluation.py \
# --base_model_name_or_path='bigcode/starcoderbase-1b' \
# --peft_model_path='//projects/bbvz/choprahetarth/new_experiments/experiment_1/final_checkpoint_starcoderbase-1b_lr_0.0001_bs_64_ms_54_dp_/u/choprahetarth/all_files/data/train_ftdata-new-small.json' \
# --save='/scratch/bbvz/choprahetarth/experiment_1/fused_model' \
# --batch_size=32


# srun --account=bbvz-delta-gpu \
# python -m torch.distributed.run --nproc_per_node=4 /u/choprahetarth/all_files/starcoder/external_evals/starcoder_finetuned_evaluation.py \
# --base_model_name_or_path='meta-llama/CodeLlama-7b-hf' \
# --peft_model_path='//scratch/bbvz/choprahetarth/starcoder2_script/codellama-7b-hf/final_checkpoint' \
# --save='//scratch/bbvz/choprahetarth/starcoder2_script/codellama-7b-hf' \
# --batch_size=20


# srun --account=bbvz-delta-gpu \
# python /u/choprahetarth/all_files/starcoder/external_evals/starcoder_finetuned_evaluation.py \
# --base_model_name_or_path='meta-llama/CodeLlama-7b-hf' \
# --peft_model_path='//scratch/bbvz/choprahetarth/starcoder2_script/codellama-7b-hf/final_checkpoint' \
# --save='//scratch/bbvz/choprahetarth/starcoder2_script/codellama-7b-hf' \
# --batch_size=1646

srun --account=bbvz-delta-gpu \
python /u/choprahetarth/all_files/starcoder/external_evals/starcoder_finetuned_evaluation.py \
--model_path='//scratch/bbvz/choprahetarth/merged_models/code-llama-python-ansible' \
--batch_size=1646