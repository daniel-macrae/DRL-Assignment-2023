#!/bin/bash
#SBATCH --job-name=testingSSDLite
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=regular
#SBATCH --mem=16GB
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=d.macrae@student.rug.nl
module load Miniconda3
source activate deepRL
conda install python=3.10
conda install conda=23.5.0
conda update --all

pip install gym
pip install -r req.txt
pip install --upgrade pip setuptools==57.5.0
pip install psutil
pip install stable-baselines3
pip install gymnasium
pip install gymnasium[box2d]
pip install gymnasium[extra]
pip install gymnasium[mujoco]
pip install stable-baselines3[extra]
pip install mujoco

conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
conda install -c conda-forge python-devtools
pip install xvfb libav-tools xorg-dev libsdl2-dev swig cmake