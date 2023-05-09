#!/bin/bash
#SBATCH --job-name=testingSSDLite
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=d.macrae@student.rug.nl
module load Miniconda3
conda create -n deppRL python=3.8
source activate deppRL
pip install -r requirements.txt