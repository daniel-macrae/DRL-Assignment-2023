#!/bin/bash
#SBATCH --job-name=PPO
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=regular
#SBATCH --mem=4GB
module load Miniconda3
source activate gymEnv
python make_videos.py