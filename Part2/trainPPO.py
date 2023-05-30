import gymnasium as gym
import numpy as np
import os

from stable_baselines3 import A2C, PPO, TD3 # these are the algorithms (models) we can use
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from Callbacks import SaveOnBestTrainingRewardCallback


env_name = "BipedalWalker-v3"
modelName = "PPO_Bipedal_1"



###   TRAINING UTILS  ###
# directory to save the log files in
# Logs will be saved in log_dir/modelName.csv
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

results_filename = log_dir + modelName + "_"
# this will save the best model during training
callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir, file_name=modelName)




### ENVIRONMENT ###
# Create and wrap the environment

vec_env = make_vec_env(env_name, n_envs=16)
vec_env = VecMonitor(vec_env, results_filename)


### MAKE THE MODEL  ###
model = PPO('MlpPolicy', vec_env, verbose=0,
            n_steps = 2048,
            batch_size = 64,
            gae_lambda= 0.95,
            gamma= 0.999,
            n_epochs= 10,
            ent_coef= 0.0,
            learning_rate= 3e-4,
            clip_range= 0.18,
        )


### TRAINING ###

timesteps = 5e6
model.learn(total_timesteps=int(timesteps), callback=callback)


