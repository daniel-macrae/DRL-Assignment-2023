import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import PPO


env_id = 'BipedalWalker-v3'
video_folder = 'videos/'
video_length = 1600

for modelNumber in range(1,10 + 1):
  print(modelNumber)
  env = DummyVecEnv([lambda: gym.make(env_id)])
  # Record the video starting at the first step
  env = VecVideoRecorder(env, video_folder,
                        record_video_trigger=lambda x: x == 0, video_length=video_length,
                        name_prefix="{}_PPO_{}".format(modelNumber, env_id))

  obs = env.reset()



  model = PPO.load("optimised/PPO_Bipedal_{}.zip".format(modelNumber))


  done = False
  for _ in range(video_length + 1):
    if not done:
      action = model.predict(obs[0])
      #print(action)
      obs, _, done, _ = env.step(action)

  # Save the video
  env.close()