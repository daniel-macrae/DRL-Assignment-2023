#!/bin/bash
#SBATCH --job-name=DRLenv
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=regular
#SBATCH --mem=8000
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=d.macrae@student.rug.nl
module load Miniconda3
conda create -n gymEnv python=3.10
source activate gymEnv
pip install numpy
pip install stable-baselines3
pip install gymnasium
pip install gymnasium[box2d]