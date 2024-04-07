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
#SBATCH --cpus-per-task=20
#SBATCH --output=/projects/bbvz/choprahetarth/new_experiments/error_log/%j.%N.finetune_8xA100_4tp.out
#SBATCH --err=/projects/bbvz/choprahetarth/new_experiments/error_log/%j.%N.finetune_8xA100_4tp.err


echo "START TIME: $(date)"
# CHANGE TO CUMMULATIVELY LOG OUTPUTS
LOG_PATH="/projects/bbvz/choprahetarth/new_experiments/error_log/run_${SLURM_JOBID}.log"
export MACHINE_RANK=$SLURM_NODEID
echo $SLURM_NODEID


module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load python anaconda3_gpu  # ... or any appropriate modules
module list  # job documentation and metadata
echo "job is starting on `hostname`"

# Create and activate the conda environment
echo "y" | conda create -n hetarth_py10 python=3.10.13
conda activate hetarth_py10

# Set the HF_HOME environment variable
export HF_HOME=/projects/bbvz/choprahetarth
export WANDB_DIR=/projects/bbvz/choprahetarth  # replace with your desired path

chmod +x /u/choprahetarth/all_files/starcoder/finetune/*
srun --account=bbvz-delta-gpu /u/choprahetarth/all_files/starcoder/finetune/environment_setup.py

srun --account=bbvz-delta-gpu \
python3 -m torch.distributed.run \
--nproc_per_node=4 \
finetune/custom_fine_tune.py \
--model_path="bigcode/starcoderbase-1b" \
--subset="data/finetune" \
--split="train" \
--size_valid_set 325 \
--streaming \
--seq_length 512 \
--max_steps 48 \
--batch_size 32 \
--input_column_name="input" \
--output_column_name="output" \
--gradient_accumulation_steps 16 \
--learning_rate 1e-4 \
--lr_scheduler_type="cosine" \
--num_warmup_steps 2 \
--weight_decay 0.05 \
--output_dir="/projects/bbvz/choprahetarth/new_experiments" \
--seed=1234 \
--save_freq=8


## So the total dataset is 15246 rows
## and the batch size is 32.
## so ten percent of the dataset is (15246/32)*0.1 approx 48