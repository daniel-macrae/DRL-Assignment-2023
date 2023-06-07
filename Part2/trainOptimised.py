import gymnasium as gym
import numpy as np
import os

from stable_baselines3 import A2C, PPO, TD3 # these are the algorithms (models) we can use
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from Callbacks import SaveOnBestTrainingRewardCallback


for i in range(1,5 +1):

    env_name = "BipedalWalker-v3"
    modelName = "PPO_Bipedal_" + str(i)

    ###   TRAINING UTILS  ###
    # directory to save the log files in
    # Logs will be saved in log_dir/modelName.csv
    log_dir = "optimised/"
    os.makedirs(log_dir, exist_ok=True)

    results_filename = log_dir + modelName + "_"
    # this will save the best model during training
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, file_name=modelName)


    ### ENVIRONMENT ###
    # Create and wrap the environment

    vec_env = make_vec_env(env_name, n_envs=32)
    vec_env = VecMonitor(vec_env, results_filename)  # this is the monitor, that saves the training episode results to the csv file


    ### MAKE THE MODEL  ###
    model = PPO('MlpPolicy', vec_env, verbose=0,
                n_steps = 2048,
                batch_size = 64,
                gae_lambda= 0.9,
                gamma= 0.999,
                n_epochs= 20,
                ent_coef= 0.0,
                learning_rate= 0.0005,
                clip_range= 0.3,
            )


    ### TRAINING ###
    timesteps = 5e6
    model.learn(total_timesteps=int(timesteps), callback=callback)


#PPO_Bipedal_1__n_envs=32_n_steps=2048_batch_size=64_gae_lambda=0.9_gamma=0.999_n_epochs=20_ent_coef=0.0_learning_rate=0.0005_clip_range=0.3