#!/bin/bash
#SBATCH --job-name=joint_best
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=leon.lang@student.uva.nl

source vqvae/bin/activate

python3 -u baseline/train_game.py \
	--seed 4 \
	--iterations 30000 \
	--log-interval 10 \
	--vqvae \
	--beta 0.25 \
	--vocab-size 25 \
	--discrete_latent_number 25 \
	--discrete_latent_dimension 25 \
	--discrete_communication \
#	--gumbel_softmax
