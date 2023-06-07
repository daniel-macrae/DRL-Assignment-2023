#!/bin/bash
#SBATCH --job-name=PPO
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=regular
#SBATCH --mem=4GB
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=d.macrae@student.rug.nl
module load Miniconda3
source activate gymEnv
python trainOptimised.py