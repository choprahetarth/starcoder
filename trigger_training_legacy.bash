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
export HF_HOME=/projects/bbvz/bzd2
export WANDB_DIR=/projects/bbvz/bzd2  # replace with your desired path


srun --account=bbvz-delta-gpu python3 /u/bzd2/starcoder/finetune/environment_setup.py

# srun --account=bbvz-delta-gpu python3 -m torch.distributed.run --nproc_per_node=4 finetune/custom_fine_tune.py  --model_path="bigcode/starcoderbase-1b"  --subset="data/finetune"  --split="train"  --size_valid_set 3000  --streaming  --seq_length 512  --max_steps 8937  --batch_size 3  --input_column_name="input"  --output_column_name="output" --gradient_accumulation_steps 16 --learning_rate 1e-4 --lr_scheduler_type="cosine" --num_warmup_steps 2 --weight_decay 0.05 --output_dir="./checkpoints" 
#4 hours....
# srun --account=bbvz-delta-gpu python3 -m torch.distributed.run --nproc_per_node=4 finetune/custom_fine_tune.py  --model_path="bigcode/starcoderbase-1b"  --subset="data/finetune"  --split="train"  --size_valid_set 3000  --streaming  --seq_length 1024  --max_steps 8937  --batch_size 3  --input_column_name="input"  --output_column_name="output" --gradient_accumulation_steps 16 --learning_rate 1e-4 --lr_scheduler_type="cosine" --num_warmup_steps 2 --weight_decay 0.05 --output_dir="./checkpoints_more_context" 
# srun --account=bbvz-delta-gpu python3 -m torch.distributed.run --nproc_per_node=4 finetune/custom_fine_tune.py  --model_path="bigcode/starcoderbase-3b"  --subset="data/finetune"  --split="train"  --size_valid_set 1000  --streaming  --seq_length 512  --max_steps 5000  --batch_size 1  --input_column_name="input"  --output_column_name="output" --gradient_accumulation_steps 16 --learning_rate 1e-4 --lr_scheduler_type="cosine" --num_warmup_steps 2 --weight_decay 0.05 --output_dir="./checkpoints_3b_model" 
# srun --account=bbvz-delta-gpu python3 -m torch.distributed.run --nproc_per_node=4 finetune/custom_fine_tune.py  --model_path="bigcode/starcoderbase-3b"  --subset="data/finetune"  --split="train"  --size_valid_set 500  --streaming  --seq_length 512  --max_steps 1500  --batch_size 1  --input_column_name="input"  --output_column_name="output" --gradient_accumulation_steps 16 --learning_rate 1e-4 --lr_scheduler_type="cosine" --num_warmup_steps 2 --weight_decay 0.05 --output_dir="./checkpoints_7b_model" 
# srun --account=bbvz-delta-gpu python3 -m torch.distributed.run --nproc_per_node=4 finetune/custom_fine_tune.py  --model_path="bigcode/starcoderbase-1b"  --subset="data/finetune"  --split="train"  --size_valid_set 1000  --streaming  --seq_length 512  --max_steps 1000  --batch_size 1  --input_column_name="input"  --output_column_name="output" --gradient_accumulation_steps 16 --learning_rate 1e-4 --lr_scheduler_type="cosine" --num_warmup_steps 2 --weight_decay 0.05 --output_dir="./checkpoints_testing"

# FINAL TRAINING



# srun --account=bbvz-delta-gpu \
# python3 -m torch.distributed.run \
# --nproc_per_node=4 \
# finetune/custom_fine_tune.py \
# --model_path="bigcode/starcoderbase-1b" \
# --subset="data/finetune" \
# --split="train" \
# --size_valid_set 1300 \
# --streaming \
# --seq_length 512 \
# --max_steps 13000 \
# --batch_size 1 \
# --input_column_name="input" \
# --output_column_name="output" \
# --gradient_accumulation_steps 16 \
# --learning_rate 1e-4 \
# --lr_scheduler_type="cosine" \
# --num_warmup_steps 2 \
# --weight_decay 0.05 \
# --output_dir="/projects/bbvz/bzd2/checkpoints_experiment_1" \
# --seed=1234 
# --save_freq=50
# tail -f /u/bzd2/starcoder/slurm-2748375.out



# srun --account=bbvz-delta-gpu \
# python3 -m torch.distributed.run \
# --nproc_per_node=4 \
# finetune/custom_fine_tune.py \
# --model_path="bigcode/starcoderbase-1b" \
# --subset="data/finetune" \
# --split="train" \
# --size_valid_set 650 \
# --streaming \
# --seq_length 512 \
# --max_steps 6500 \
# --batch_size 1 \
# --input_column_name="input" \
# --output_column_name="output" \
# --gradient_accumulation_steps 16 \
# --learning_rate 1e-4 \
# --lr_scheduler_type="cosine" \
# --num_warmup_steps 2 \
# --weight_decay 0.05 \
# --output_dir="/projects/bbvz/bzd2/checkpoints_experiment_3" \
# --seed=1234 
# --save_freq=50
# tail -f /u/bzd2/starcoder/slurm-2748376.out


# srun --account=bbvz-delta-gpu \
# python3 -m torch.distributed.run \
# --nproc_per_node=4 \
# finetune/custom_fine_tune.py \
# --model_path="bigcode/starcoderbase-1b" \
# --subset="data/finetune" \
# --split="train" \
# --size_valid_set 325 \
# --streaming \
# --seq_length 512 \
# --max_steps 3250 \
# --batch_size 1 \
# --input_column_name="input" \
# --output_column_name="output" \
# --gradient_accumulation_steps 16 \
# --learning_rate 1e-4 \
# --lr_scheduler_type="cosine" \
# --num_warmup_steps 2 \
# --weight_decay 0.05 \
# --output_dir="/projects/bbvz/bzd2/checkpoints_experiment_4" \
# --seed=1234 
# --save_freq=50
# tail -f /u/bzd2/starcoder/slurm-2748380.out



# srun --account=bbvz-delta-gpu \
# python3 -m torch.distributed.run \
# --nproc_per_node=4 \
# finetune/custom_fine_tune.py \
# --model_path="bigcode/starcoderbase-1b" \
# --subset="data/finetune" \
# --split="train" \
# --size_valid_set 1300 \
# --streaming \
# --seq_length 1024 \
# --max_steps 13000 \
# --batch_size 1 \
# --input_column_name="input" \
# --output_column_name="output" \
# --gradient_accumulation_steps 16 \
# --learning_rate 1e-4 \
# --lr_scheduler_type="cosine" \
# --num_warmup_steps 2 \
# --weight_decay 0.05 \
# --output_dir="/projects/bbvz/bzd2/checkpoints_experiment_5" \
# --seed=1234 
# --save_freq=50
# tail -f /u/bzd2/starcoder/slurm-2748382.out


# srun --account=bbvz-delta-gpu \
# python3 -m torch.distributed.run \
# --nproc_per_node=4 \
# finetune/custom_fine_tune.py \
# --model_path="bigcode/starcoderbase-1b" \
# --subset="data/finetune" \
# --split="train" \
# --size_valid_set 650 \
# --streaming \
# --seq_length 1024 \
# --max_steps 6500 \
# --batch_size 1 \
# --input_column_name="input" \
# --output_column_name="output" \
# --gradient_accumulation_steps 16 \
# --learning_rate 1e-4 \
# --lr_scheduler_type="cosine" \
# --num_warmup_steps 2 \
# --weight_decay 0.05 \
# --output_dir="/projects/bbvz/bzd2/checkpoints_experiment_6" \
# --seed=1234 
# --save_freq=50
# tail -f /u/bzd2/starcoder/slurm-2748383.out


# srun --account=bbvz-delta-gpu \
# python3 -m torch.distributed.run \
# --nproc_per_node=4 \
# finetune/custom_fine_tune.py \
# --model_path="bigcode/starcoderbase-3b" \
# --subset="data/finetune" \
# --split="train" \
# --size_valid_set 650 \
# --streaming \
# --seq_length 512 \
# --max_steps 6500 \
# --batch_size 1 \
# --input_column_name="input" \
# --output_column_name="output" \
# --gradient_accumulation_steps 16 \
# --learning_rate 1e-4 \
# --lr_scheduler_type="cosine" \
# --num_warmup_steps 2 \
# --weight_decay 0.05 \
# --output_dir="/projects/bbvz/bzd2/checkpoints_experiment_7_A40" \
# --seed=1234 
# --save_freq=50
# tail -f /u/bzd2/starcoder/slurm-2748384.out



# srun --account=bbvz-delta-gpu \
# python3 -m torch.distributed.run \
# --nproc_per_node=4 \
# finetune/custom_fine_tune.py \
# --model_path="bigcode/starcoderbase-3b" \
# --subset="data/finetune" \
# --split="train" \
# --size_valid_set 325 \
# --streaming \
# --seq_length 512 \
# --max_steps 3250 \
# --batch_size 1 \
# --input_column_name="input" \
# --output_column_name="output" \
# --gradient_accumulation_steps 16 \
# --learning_rate 1e-4 \
# --lr_scheduler_type="cosine" \
# --num_warmup_steps 2 \
# --weight_decay 0.05 \
# --output_dir="/projects/bbvz/bzd2/checkpoints_experiment_8_A100" \
# --seed=1234 
# --save_freq=50
# tail -f /u/bzd2/starcoder/slurm-2748385.out



#### NEW DATASET TOP 10% 

# srun --account=bbvz-delta-gpu \
# python3 -m torch.distributed.run \
# --nproc_per_node=4 \
# finetune/custom_fine_tune.py \
# --model_path="bigcode/starcoderbase-3b" \
# --subset="data/finetune" \
# --split="train" \
# --size_valid_set 650 \
# --streaming \
# --seq_length 512 \
# --max_steps 6500 \
# --batch_size 1 \
# --input_column_name="input" \
# --output_column_name="output" \
# --gradient_accumulation_steps 16 \
# --learning_rate 1e-4 \
# --lr_scheduler_type="cosine" \
# --num_warmup_steps 2 \
# --weight_decay 0.05 \
# --output_dir="/projects/bbvz/bzd2/checkpoints_experiment_9" \
# --seed=1234 
# --save_freq=50
# tail -f /u/bzd2/starcoder/slurm-2748388.out


# srun --account=bbvz-delta-gpu \
# python3 -m torch.distributed.run \
# --nproc_per_node=4 \
# finetune/custom_fine_tune.py \
# --model_path="bigcode/starcoderbase-3b" \
# --subset="data/finetune" \
# --split="train" \
# --size_valid_set 325 \
# --streaming \
# --seq_length 512 \
# --max_steps 3250 \
# --batch_size 1 \
# --input_column_name="input" \
# --output_column_name="output" \
# --gradient_accumulation_steps 16 \
# --learning_rate 1e-4 \
# --lr_scheduler_type="cosine" \
# --num_warmup_steps 2 \
# --weight_decay 0.05 \
# --output_dir="/projects/bbvz/bzd2/checkpoints_experiment_10" \
# --seed=1234 
# --save_freq=50
# tail -f /u/bzd2/starcoder/slurm-2748389.out




# srun --account=bbvz-delta-gpu \
# python3 -m torch.distributed.run \
# --nproc_per_node=4 \
# finetune/custom_fine_tune.py \
# --model_path="bigcode/starcoderbase-1b" \
# --subset="data/finetune" \
# --split="train" \
# --size_valid_set 1300 \
# --streaming \
# --seq_length 512 \
# --max_steps 13000 \
# --batch_size 1 \
# --input_column_name="input" \
# --output_column_name="output" \
# --gradient_accumulation_steps 16 \
# --learning_rate 1e-4 \
# --lr_scheduler_type="cosine" \
# --num_warmup_steps 2 \
# --weight_decay 0.05 \
# --output_dir="/projects/bbvz/bzd2/checkpoints_experiment_2" \
# --seed=1234 
# --save_freq=50
# tail -f /u/bzd2/starcoder/slurm-2748391.out


# srun --account=bbvz-delta-gpu \
# python3 -m torch.distributed.run \
# --nproc_per_node=4 \
# finetune/custom_fine_tune.py \
# --model_path="bigcode/starcoderbase-7b" \
# --subset="data/finetune" \
# --split="train" \
# --size_valid_set 650 \
# --streaming \
# --seq_length 512 \
# --max_steps 6500 \
# --batch_size 1 \
# --input_column_name="input" \
# --output_column_name="output" \
# --gradient_accumulation_steps 16 \
# --learning_rate 1e-4 \
# --lr_scheduler_type="cosine" \
# --num_warmup_steps 2 \
# --weight_decay 0.05 \
# --output_dir="/projects/bbvz/bzd2/checkpoints_experiment_7b_5_percent" \
# --seed=1234 
# --save_freq=50
# # tail -f /u/bzd2/starcoder/slurm-2890665.out

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