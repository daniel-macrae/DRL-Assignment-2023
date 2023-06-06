#!/bin/bash
#SBATCH --job-name=DRLenv
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=2000
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=d.macrae@student.rug.nl
module load Miniconda3
conda create -n gymEnv python=3.10
source activate deepRL
pip install numpy
pip install stable-baselines3
pip install gymnasium
pip install gymnasium[box2d]