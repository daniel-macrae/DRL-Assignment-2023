import gymnasium as gym

env = gym.make("Ant-v4", render_mode = 'rgb_array')

print('this didnt crash')