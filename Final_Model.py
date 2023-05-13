from agent_and_memory import Agent, MemoryBuffer
from Catch import CatchEnv
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## DQN Parameters
params = {'batch_size' : 128,
                'gamma' : 0.99,
                'eps_start' : 0.9,
                'eps_end' : 0,
                'eps_decay' : 500,
                'learning_rate' : 1e-3,
                'memory_buffer' : 50000,
                'ams_grad' : True,
                'targetnet_update_rate' : 10}

if torch.cuda.is_available():
    num_episodes = 5000
else:
    num_episodes = 50


# function to perform the 10 episodes of testing the agent, without any learning taking place
def testingTenEpisodes(agent, env):
    total_reward = 0
    for episode in range(10):        

        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        state = state.permute(2, 0, 1).unsqueeze(0) 

        terminal = False
        
        while not terminal:
            action = agent.select_action(state, testing = True)     # testing=True enforces exploitation, no exploration takes place
            next_state, reward, terminal = env.step(action.item()) 

            if not terminal:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
                next_state = next_state.permute(2, 0, 1).unsqueeze(0)
            
            if terminal:
                next_state = None

            total_reward += reward
            state = next_state


    return total_reward / 10




env = CatchEnv()
num_moves = env.get_num_actions()


for runNumber in range(1,6): 
    print("Run number:", runNumber)
    evaluation_results = [] 

    # initialise the memory buffer and agent
    memoryBuffer = MemoryBuffer(params['memory_buffer'])
    agent = Agent(num_moves, params['eps_start'], params['eps_end'], params['eps_decay'], memoryBuffer, params['batch_size'], params['learning_rate'], params['ams_grad'], params['gamma'], params['targetnet_update_rate'])


    for episode in range(num_episodes):
        agent.episode += 1

        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) 

        terminal = False

        while not terminal:
            # agent interacts with the environment
            action = agent.select_action(state)    
            next_state, reward, terminal = env.step(action.item()) 
            
            # turn everything into tensors here, before putting in memory
            reward = torch.tensor([reward], device=device)
            if not terminal:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            else:
                next_state = None
                            
            # add trajectory to memory buffer and move to the next state
            agent.memory.push(state, action, next_state, reward)
            state = next_state

            # optimise the DQN model
            agent.optimize_model()

        # testing of the agent between 10 episode blocks
        if episode % 10 == 0 and episode > 0:
            score = testingTenEpisodes(agent, env)
            evaluation_results.append(score)
        
    #print(evaluation_results)
    evaluation_results = np.array(evaluation_results)

    filename = "Results/group_02_catch_rewards_" + str(runNumber) + ".npy"
    np.save(filename, evaluation_results)

    del agent.model, agent.target_network
    del agent, memoryBuffer





