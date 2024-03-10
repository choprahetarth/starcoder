#!/bin/bash
#SBATCH --mem=100g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16     # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4 # <- one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account='bbvz-delta-gpu'
#SBATCH --job-name="finetune/custom_fine_tune.py"
#SBATCH --time=5:00:00
### GPU options ###
#SBATCH --gpus-per-node=4
# SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load python anaconda3_gpu  # ... or any appropriate modules
module list  # job documentation and metadata
echo "job is starting on `hostname`"
# Create and activate the conda environment
# conda create -n hetarth_py10 python=3.10.13
conda activate hetarth_py10
# Set the HF_HOME environment variable
export HF_HOME=/projects/bbvz/bzd2
export WANDB_DIR=/projects/bbvz/bzd2  # replace with your desired path
# srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder/finetune/environment_setup.py
# srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder_evaluation.py --base_model_name_or_path='bigcode/starcoderbase-1b' --peft_model_path='/projects/bbvz/bzd2/checkpoints_experiment_4/final_checkpoint' --save='/projects/bbvz/bzd2/checkpoints_experiment_4/fused_model'
# srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder_evaluation.py --base_model_name_or_path='bigcode/starcoderbase-3b' --peft_model_path='/projects/bbvz/bzd2/checkpoints_experiment_10/final_checkpoint' --save='/projects/bbvz/bzd2/checkpoints_experiment_10/fused_model'
# srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder_evaluation.py --base_model_name_or_path='bigcode/starcoderbase-3b' --peft_model_path='/projects/bbvz/bzd2/checkpoints_experiment_8_A100/final_checkpoint' --save='/projects/bbvz/bzd2/checkpoints_experiment_8_A100/fused_model'
# srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder_evaluation.py --base_model_name_or_path='bigcode/starcoderbase-1b' --peft_model_path='/projects/bbvz/bzd2/checkpoints_experiment_6/final_checkpoint' --save='/projects/bbvz/bzd2/checkpoints_experiment_6/fused_model'
# srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder_evaluation.py --base_model_name_or_path='bigcode/starcoderbase-1b' --peft_model_path='/projects/bbvz/bzd2/checkpoints_experiment_3/final_checkpoint' --save='/projects/bbvz/bzd2/checkpoints_experiment_3/fused_model'
# srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder_evaluation.py --base_model_name_or_path='bigcode/starcoderbase-1b' --peft_model_path='/projects/bbvz/bzd2/checkpoints_experiment_1/final_checkpoint' --save='/projects/bbvz/bzd2/checkpoints_experiment_1/fused_model'
# srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder_evaluation.py --base_model_name_or_path='bigcode/starcoderbase-1b' --peft_model_path='/projects/bbvz/bzd2/checkpoints_experiment_2/final_checkpoint' --save='/projects/bbvz/bzd2/checkpoints_experiment_2/fused_model'
# srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder_evaluation.py --base_model_name_or_path='bigcode/starcoderbase-3b' --peft_model_path='/projects/bbvz/bzd2/checkpoints_experiment_9/final_checkpoint' --save='/projects/bbvz/bzd2/checkpoints_experiment_9/fused_model'
# srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder_evaluation.py --base_model_name_or_path='bigcode/starcoderbase-3b' --peft_model_path='/projects/bbvz/bzd2/checkpoints_experiment_7_A40/final_checkpoint' --save='/projects/bbvz/bzd2/checkpoints_experiment_7_A40/fused_model'
# srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder_evaluation.py --base_model_name_or_path='bigcode/starcoderbase-1b' --peft_model_path='/projects/bbvz/bzd2/checkpoints_experiment_5/final_checkpoint' --save='/projects/bbvz/bzd2/checkpoints_experiment_5/fused_model'
# srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder_evaluation.py --base_model_name_or_path='bigcode/starcoderbase-7b' --peft_model_path='/projects/bbvz/bzd2/checkpoints_experiment_7b/final_checkpoint' --save='/projects/bbvz/bzd2/checkpoints_experiment_7b/fused_model'

# srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder_base_eval.py

# 
# last two are 1 and 3 b respectively
# 2 , 9 , 10, 1, 3, 4, 6, 7, 8



