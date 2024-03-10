#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16     # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4  # <- one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=account_name
#SBATCH --job-name=finetune
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load python  # ... or any appropriate modules
module list  # job documentation and metadata
echo "job is starting on `hostname`"

srun python3 -m torch.distributed.run --nproc_per_node=1 finetune/custom_fine_tune.py --model_path="bigcode/starcoderbase-1b" --dataset_name="ArmelR/stack-exchange-instruction" --subset="data/finetune" --split="train" --size_valid_set 500 --streaming --seq_length 2048 --max_steps 1000 --batch_size 1 --input_column_name="input" --output_column_name="output" --gradient_accumulation_steps 16 --learning_rate 1e-4 --lr_scheduler_type="cosine" --num_warmup_steps 100 --weight_decay 0.05
