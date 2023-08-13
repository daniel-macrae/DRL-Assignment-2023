#!/bin/bash
#SBATCH --job-name=PPO
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=regular
#SBATCH --mem=4GB
module load Miniconda3
source activate gymEnv
python gridsearch.py --n_envs 16