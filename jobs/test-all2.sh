#!/bin/bash
#SBATCH --job-name=joint_best
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=leon.lang@student.uva.nl


source vqvae/bin/activate


### Now: No RL anymore ###
# ultimate baseline:
python3 -u baseline/train_game.py \
	--seed 1 \
	--iterations 1000 \
	--log-interval 10

# from now on: no rl, but vqvae
# continuous communication:
python3 -u baseline/train_game.py \
	--seed 1 \
	--iterations 1000 \
	--log-interval 10 \
	--vqvae

# discrete communication, but no gumbel softmax:
python3 -u baseline/train_game.py \
	--seed 1 \
	--iterations 1000 \
	--log-interval 10
	--vqvae \
	--discrete_communication

# discrete communication, and gumbel softmax:
python3 -u baseline/train_game.py \
	--seed 1 \
	--iterations 1000 \
	--log-interval 10
	--vqvae \
	--discrete_communication \
	--gumbel_softmax

