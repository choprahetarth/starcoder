#!/bin/bash
#SBATCH --mem=100g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16     # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4 # <- one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account='bbvz-delta-gpu'
#SBATCH --job-name="finetune/custom_fine_tune.py"
#SBATCH --time=30:00:00
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
conda create -n hetarth_py10 python=3.10.13
conda activate hetarth_py10

# Set the HF_HOME environment variable
export HF_HOME=/projects/bbvz/choprahetarth
export WANDB_DIR=/projects/bbvz/choprahetarth  # replace with your desired path


srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder/finetune/environment_setup.py

srun --account=bbvz-delta-gpu \
python3 -m torch.distributed.run \
--nproc_per_node=4 \
finetune/custom_fine_tune.py \
--model_path="bigcode/starcoderbase-7b" \
--subset="data/finetune" \
--split="train" \
--size_valid_set 325 \
--streaming \
--seq_length 512 \
--max_steps 3250 \
--batch_size 1 \
--input_column_name="input" \
--output_column_name="output" \
--gradient_accumulation_steps 16 \
--learning_rate 1e-4 \
--lr_scheduler_type="cosine" \
--num_warmup_steps 2 \
--weight_decay 0.05 \
--output_dir="/projects/bbvz/bzd2/checkpoints_experiment_7b_2_5_percent" \
--seed=1234 
--save_freq=50
# tail -f /u/bzd2/starcoder/slurm-2891635.out