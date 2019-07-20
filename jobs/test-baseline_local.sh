#!/bin/bash
#SBATCH --job-name=joint_best
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#module purge
#module load eb
#module load Python/3.6.3-foss-2017b
#module load cuDNN/7.0.5-CUDA-9.0.176
#module load NCCL/2.0.5-CUDA-9.0.176
#export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

python3 -u baseline/train_game.py \
	--device cpu \
	--seed 114 \
	--iterations 200 \
	--log-interval 10 \
	--vocab-size 25 \
