import gym
import numpy as np
import os
from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from Callbacks import SaveOnBestTrainingRewardCallback

#env_name = "BipedalWalker-v3"
#modelName = "PPO_Bipedal_1"


def create_log_file(log_file):
    # Check if the file already exists
    if not os.path.exists(log_file):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create the file
        with open(log_file, 'w') as file:
            pass  # Do nothing, just create an empty file


env_name = "CliffWalking-v0"
modelName = "PPO_CliffWalking_1"

log_dir = "tmp/gridsearch/"
os.makedirs(log_dir, exist_ok=True)
results_filename = log_dir + modelName + "_"


# Define the parameter combinations to test
n_steps_values = [1024]#, 2048, 4096]
batch_size_values = [32, 64]#, 128]
gae_lambda_values = [0.9]#, 0.95, 0.99]
gamma_values = [0.99]#, 0.995, 0.999]
n_epochs_values = [5]#, 10, 20]
ent_coef_values = [0.0]#, 0.01, 0.1]
learning_rate_values = [1e-4]#, 3e-4, 1e-3]
clip_range_values = [0.1]#, 0.5, 0.3]

# Iterate over all parameter combinations
for n_steps in n_steps_values:
    for batch_size in batch_size_values:
        for gae_lambda in gae_lambda_values:
            for gamma in gamma_values:
                for n_epochs in n_epochs_values:
                    for ent_coef in ent_coef_values:
                        for learning_rate in learning_rate_values:
                            for clip_range in clip_range_values:
                                vec_env = make_vec_env(env_name, n_envs=16)
                                vec_env = VecMonitor(vec_env, results_filename)

                                model = PPO('MlpPolicy', vec_env, verbose=0,
                                            n_steps=n_steps,
                                            batch_size=batch_size,
                                            gae_lambda=gae_lambda,
                                            gamma=gamma,
                                            n_epochs=n_epochs,
                                            ent_coef=ent_coef,
                                            learning_rate=learning_rate,
                                            clip_range=clip_range)
                                
                                log_file_name = modelName + "_" + f"n_steps={n_steps}_batch_size={batch_size}_gae_lambda={gae_lambda}_gamma={gamma}_n_epochs={n_epochs}_ent_coef={ent_coef}_learning_rate={learning_rate}_clip_range={clip_range}"
                                callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, file_name=log_file_name)
                                timesteps = 5e6
                                model.learn(total_timesteps=int(timesteps), callback=callback)

# Extract the rewards from the log files
rewards = []
for n_steps in n_steps_values:
    for batch_size in batch_size_values:
        for gae_lambda in gae_lambda_values:
            for gamma in gamma_values:
                for n_epochs in n_epochs_values:
                    for ent_coef in ent_coef_values:
                        for learning_rate in learning_rate_values:
                            for clip_range in clip_range_values:
                                log_file = log_dir + modelName + "_" + f"n_steps={n_steps}_batch_size={batch_size}_gae_lambda={gae_lambda}_gamma={gamma}_n_epochs={n_epochs}_ent_coef={ent_coef}_learning_rate={learning_rate}_clip_range={clip_range}.csv"
                                rewards.append(np.loadtxt(log_file, delimiter=",", skiprows=1, usecols=1)[-1])

# Find the index of the best reward
best_reward_index = np.argmax(rewards)

# Retrieve the corresponding best parameter combination
best_n_steps = n_steps_values[best_reward_index]
best_batch_size = batch_size_values[best_reward_index]
best_gae_lambda = gae_lambda_values[best_reward_index]
best_gamma = gamma_values[best_reward_index]
best_n_epochs = n_epochs_values[best_reward_index]
best_ent_coef = ent_coef_values[best_reward_index]
best_learning_rate = learning_rate_values[best_reward_index]
best_clip_range = clip_range_values[best_reward_index]

# Print the best parameter combination and its corresponding reward
print("Best Parameter Combination:")
print(f"n_steps: {best_n_steps}")
print(f"batch_size: {best_batch_size}")
print(f"gae_lambda: {best_gae_lambda}")
print(f"gamma: {best_gamma}")
print(f"n_epochs: {best_n_epochs}")
print(f"ent_coef: {best_ent_coef}")
print(f"learning_rate: {best_learning_rate}")
print(f"clip_range: {best_clip_range}")
print(f"Best Reward: {rewards[best_reward_index]}")