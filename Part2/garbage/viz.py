import gymnasium as gym
import numpy as np
import imageio


# ALSO GOOD:
# https://stable-baselines.readthedocs.io/en/master/guide/examples.html#bonus-make-a-gif-of-a-trained-agent


#
folder = "gifs/"

# Make the environment and the initial observation 
env_name = 'Walker2d-v4'
env = gym.make(env_name, render_mode='rgb_array')
obs_dim = env.observation_space.shape
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]
observation = env.reset()

# Book-keeping.
num_episodes = 1 # how many gifs to make
ep_return = 0
ep_length = 0
ep_done = 0
ep_observations = []

# For video / GIF.
dur = 0.001
#width = 250
#height = 200

while ep_done < num_episodes:
    observation = env.render()

    #assert obs.shape == (height, width, 3), obs.shape  # height first!
    ep_observations.append(observation)

    # Take action, step into environment, etc.
    """
    CHANGE THE NEXT LINE (action selection model)
    """
    action = env.action_space.sample()  # Replace with your own action selection logic


    (observation, reward, done, something2, _) = env.step(action)

    ep_return += reward
    ep_length += 1

    if done:
        # Form GIF. imageio should read from numpy: https://imageio.github.io/
        print(f'Episode {ep_done}, cum. return: {ep_return:0.1f}, length: {ep_length}.')
        ep_name = folder + f'ep_{env_name}_{str(ep_done).zfill(2)}_dur_{dur}_len_{str(ep_length).zfill(3)}.gif'
        with imageio.get_writer(ep_name, mode='I', duration=dur) as writer:
            for obs_np in ep_observations:
                writer.append_data(obs_np)

        # Reset information.
        observation = env.reset()
        ep_ret = 0
        ep_len = 0
        ep_done += 1
        ep_obs = []