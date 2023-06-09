import numpy as np
import os
import re

#r: Represents the reward value obtained during an episode of the training process.
#l: Indicates the number of steps taken during that episode.
#t: Represents the elapsed time (in seconds) during that episode.

log_dir = "gridsearch"  
output_file_path = "grid_eval/average_rewards.csv"

average_rewards = []

print("Average Reward over the last 1000 episodes:")

for log_file in os.listdir(log_dir):
    log_file_path = os.path.join(log_dir, log_file)
    # checking if it is a file
    if os.path.isfile(log_file_path):
        if log_file.startswith(".nfs"):
            continue  # Skip files starting with ".nfs"
       
        # Load the log file and extract the reward values
        rewards = []
        with open(log_file_path, "r") as log_file:
            for line in log_file:
                if not line.startswith("#"):  # Skip comment lines
                    try:
                        #print(line)
                        #time = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1])  # Extract the last float number
                        reward = float(line.split(",")[0])  # Extract the reward value
    
                        rewards.append(reward)
                    except:
                        print("could not turn to float: ", line.split(",")[0])
                        print("in file: ", log_file_path )
                        continue

        # Calculate the average reward over the last 1000 episodes
        last_1000_rewards = rewards[-1000:]
        average_reward = np.mean(last_1000_rewards)
        print("average reward is: ", average_reward, " , for", log_file.name.strip())

        # Save the average reward and corresponding file name to the output file
        with open(output_file_path, "a") as output_file:
            file_name = os.path.basename(log_file.name.strip())
            output_file.write(f"{file_name}: {average_reward}\n")

        average_rewards.append((file_name, average_reward))  # Save the values in a list for later use

print("Average rewards saved to:", output_file_path)


max_average_reward = float("-inf")
max_average_reward_file = ""

with open(output_file_path, "r") as output_file:
    for line in output_file:
        try:
            file_name, average_reward = line.strip().split(": ")
        except:
            print(line)
        average_reward = float(average_reward)

        if average_reward > max_average_reward:
            max_average_reward = average_reward
            max_average_reward_file = file_name

print("Maximum Average Reward:", max_average_reward)
print("File Name:", os.path.basename(max_average_reward_file))


# 300 points in 1600 time steps.