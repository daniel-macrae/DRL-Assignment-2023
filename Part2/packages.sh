#!/bin/bash
#SBATCH --job-name=PPO
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=regular
#SBATCH --mem=2GB
module load Miniconda3
source activate gymEnv
conda list -n gymEnv