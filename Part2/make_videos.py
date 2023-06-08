import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import PPO


env_id = 'BipedalWalker-v3'
video_folder = 'videos/'
video_length = 500


env = DummyVecEnv([lambda: gym.make(env_id)])

obs = env.reset()

# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix="1_PPO_{}".format(env_id))

obs = env.reset()


loadedModel = PPO('MlpPolicy', env, verbose=1)
loadedModel = loadedModel.load("optimised/PPO_Bipedal_1.zip")

for _ in range(video_length + 1):
  action = loadedModel.predict(obs)
  obs, _, _, _ = env.step(action)

# Save the video
env.close()