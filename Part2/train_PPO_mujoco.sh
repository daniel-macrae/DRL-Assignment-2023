#!/bin/bash
#SBATCH --job-name=PPO
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=regular
#SBATCH --mem=8GB
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=d.macrae@student.rug.nl
module load Miniconda3
module load foss/2022a
source activate deepRL
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home1/s3719332/.mujoco/mujoco210/bin
export CPATH=$CONDA_PREFIX/include
python trainPPO.py