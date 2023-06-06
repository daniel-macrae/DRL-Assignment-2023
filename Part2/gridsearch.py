import gym
import numpy as np
import os
from stable_baselines3.common.env_util import make_vec_env
from sklearn.model_selection import GridSearchCV

from stable_baselines3.common.envs import DummyVecEnv

class StableBaselinesWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, policy, env_name, **kwargs):
        self.policy = policy
        self.env_name = env_name
        self.model = None
    
    def fit(self, X, y=None):
        env = DummyVecEnv([lambda: gym.make(self.env_name)])
        self.model = PPO(self.policy, env, **kwargs)
        self.model.learn(total_timesteps=len(X))
        return self
    
    def predict(self, X, y=None):
        return self.model.predict(X)
env_name = "Pendulum-v1"

# Create the environment
env = gym.make(env_name)

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

# Create the PPO model
#model = PPO('MlpPolicy', env)

# Create the PPO model wrapper
model = StableBaselinesWrapper('MlpPolicy', env_name)

# Create the grid search object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

# Fit the grid search object to perform the search
grid_search.fit(env)

# Print the best parameters and score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
