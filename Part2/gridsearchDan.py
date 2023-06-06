import gym
import numpy as np
import os
from sklearn.model_selection import ParameterGrid
import random

from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from Callbacks import SaveOnBestTrainingRewardCallback



env_name = "CliffWalking-v0"
modelName = "PPO_CliffWalking_1"

#env_name = "BipedalWalker-v3"
#modelName = "PPO_Bipedal_1"

log_dir = "gridsearch/"
os.makedirs(log_dir, exist_ok=True)
model_filename_base = log_dir + modelName + "_"


# Define the parameter combinations to test

param_grid = {
    'n_envs' : [8, 16, 32],
    'n_steps': [1024, 2048, 4096],
    'batch_size': [32, 64, 128],
    'gae_lambda': [0.9, 0.95, 0.99],
    'gamma': [0.8, 0.9, 0.999],
    'n_epochs': [5, 10, 20],
    'ent_coef': [0.0, 0.1],
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'clip_range': [0.1, 0.2, 0.3]
}

grid = list(ParameterGrid(param_grid))
random.shuffle(grid)

print(len(grid))

for params in grid:
    n_envs = params['n_envs']
    n_steps = params['n_steps']
    batch_size = params['batch_size']
    gae_lambda  = params['gae_lambda']
    gamma = params['gamma']
    n_epochs = params['n_epochs']
    ent_coef = params['ent_coef']
    learning_rate = params['learning_rate']
    clip_range = params['clip_range']

    model_filename = modelName + f"_n_envs={n_envs}_n_steps={n_steps}_batch_size={batch_size}_gae_lambda={gae_lambda}_gamma={gamma}_n_epochs={n_epochs}_ent_coef={ent_coef}_learning_rate={learning_rate}_clip_range={clip_range}"
    print(model_filename)
    #callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir, file_name=model_filename, verbose=0)

    log_file_name = model_filename_base + f"_n_envs={n_envs}_n_steps={n_steps}_batch_size={batch_size}_gae_lambda={gae_lambda}_gamma={gamma}_n_epochs={n_epochs}_ent_coef={ent_coef}_learning_rate={learning_rate}_clip_range={clip_range}_"
    vec_env = make_vec_env(env_name, n_envs=n_envs)
    vec_env = VecMonitor(vec_env, log_file_name)

    model = PPO('MlpPolicy', vec_env, verbose=0,
                n_steps=n_steps,
                batch_size=batch_size,
                gae_lambda=gae_lambda,
                gamma=gamma,
                n_epochs=n_epochs,
                ent_coef=ent_coef,
                learning_rate=learning_rate,
                clip_range=clip_range)

    timesteps = 4e6
    model.learn(total_timesteps=int(timesteps))#, callback=callback)



