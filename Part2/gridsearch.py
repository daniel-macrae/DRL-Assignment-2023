from sklearn.model_selection import GridSearchCV
#from stable_baselines3.common.envs import VecNormalize
from Callbacks import SaveOnBestTrainingRewardCallback
import gymnasium as gym
import numpy as np
import os

from stable_baselines3 import A2C, PPO, TD3 # these are the algorithms (models) we can use
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from Callbacks import SaveOnBestTrainingRewardCallback

env_name = "CliffWalking-v0"
modelName = "PPO_CliffWalking_1"

env = gym.make(env_name)

###   TRAINING UTILS  ###
# directory to save the log files in
# Logs will be saved in log_dir/modelName.csv
log_dir = "tmp/gridsearch/"
os.makedirs(log_dir, exist_ok=True)

results_filename = log_dir + modelName + "_"

# Define the parameter grid
param_grid = {
    'n_steps': [1024], #, 2048, 4096],
    'batch_size': [32], #, 64, 128],
    'gae_lambda': [0.9], #, 0.95, 0.99],
    'gamma': [0.99], #, 0.999, 0.995],
    'n_epochs': [5], #, 10, 20],
    'ent_coef': [0.0], #, 0.0, 0.1],
    'learning_rate': [1e-4], #, 5e-4, 1e-3],
    'clip_range': [0.1], #, 0.2, 0.3]
}


# Define the SaveOnBestTrainingRewardCallback
callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir, file_name=modelName)

# Define the function for training and evaluation
# Define the function for training and evaluation
def train_model(params):
    timesteps = 5e6

    vec_env = make_vec_env(env_name, n_envs=16)
    vec_env = VecMonitor(vec_env, results_filename)
    model = PPO('MlpPolicy', vec_env, verbose=0, **params)
    model.learn(total_timesteps=int(timesteps), callback=callback)
    mean_reward = callback.best_mean_reward  # Retrieve the best mean reward from the callback
    
    return mean_reward

# Create an instance of GridSearchCV with the training function, parameter grid, and scoring
grid_search = GridSearchCV(estimator=None, param_grid=param_grid, scoring='neg_mean_squared_error')
grid_search.fit(X=None, y=None, groups=None)

# Access the best hyperparameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(best_params)
print(best_model)