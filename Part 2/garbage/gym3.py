import gymnasium as gym
import numpy as np


from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import MlpPolicy


from stable_baselines3.common.evaluation import evaluate_policy









def evaluate(model, num_episodes=50, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    vec_env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = vec_env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            # also note that the step only returns a 4-tuple, as the env that is returned
            # by model.get_env() is an sb3 vecenv that wraps the >v0.26 API
            obs, reward, done, info = vec_env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "std reward:", std_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward




if __name__ == "__main__":
    env_name = "Ant-v2"
    #env = gym.make(env_name, render_mode="rgb_array")
    #model = (MlpPolicy, env, verbose=0)
    model = A2C("MlpPolicy", env_name)#.learn(100)
    #eval_env = gym.make(env_name, render_mode="rgb_array")


    for i in range(10):
        model.learn(100)
        mean_reward = evaluate(model)

    model.save("firstModel")
        #print(mean_reward)
    print("done!")