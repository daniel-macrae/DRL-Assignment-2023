#!/bin/bash
#SBATCH --job-name=testingSSDLite
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=d.macrae@student.rug.nl
module load Miniconda3
source activate deepRL
python grid_search.py