from sklearn.model_selection import ParameterGrid
import torch
import random
import pandas as pd

# import code bits
from agent_and_memory import Agent, MemoryBuffer
from Catch import CatchEnv

import argparse


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
 
    parser.add_argument("--DQN", default=0, type=int,help="version of the DQN")
    
    parser.add_argument("--filename", default="grid_search", type=str, help="name of file to store the grid search results in")

    return parser




def grid_search(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    BATCH_SIZE = [16, 32, 64, 128]
    GAMMA = [0.6, 0.8, 0.9, 0.99]
    EPS_START = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    EPS_END = [0, 0.01, 0.05]
    EPS_DECAY = [300, 500, 700, 900]
    LR = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    MEMORYBUFFER = [5000, 10000, 50000]
    AMSGRAD = [True, False]
    TARGETNET_UPDATE_RATE = [1, 5, 10, 20]


    hyper_grid = {'batch_size' : BATCH_SIZE,
                'gamma' : GAMMA,
                'eps_start' : EPS_START,
                'eps_end' : EPS_END,
                'eps_decay' : EPS_DECAY,
                'learning_rate' : LR,
                'memory_buffer' : MEMORYBUFFER,
                'ams_grad' : AMSGRAD,
                'targetnet_update_rate' : TARGETNET_UPDATE_RATE}

    grid = list(ParameterGrid(hyper_grid))
    random.shuffle(grid)  # randomly shuffle the grid (in case we don't get many trials done, at least there is more variety)
    print(len(grid))


    # AMOUNT OF GRID TO SAMPLE
    gridSampleSize = 0.7
    param_columns = list(grid[0].keys())

    # stuffs
    DQN_model = int(args.DQN)   # in case we design more models, we'll call the one we have now the 1st one
    output_filename = str(args.filename) + ".xlsx"
    print(output_filename)


    if torch.cuda.is_available():
        num_episodes = 4000
    else:
        num_episodes = 50



    # GRID SEARCH LOOP

    env = CatchEnv()
    num_moves = env.get_num_actions()


    RESULTS_DATAFRAME = pd.DataFrame(columns=["DQN_model",'batch_size', 'gamma', 'eps_start', 'eps_end', 'eps_decay', 'learning_rate', 'memory_buffer', 'ams_grad', 'targetnet_update_rate',
                                            "avgRewards", "average_last_100_episodes", "best_average_100_episodes", "time_of_peak", "time_to_convergence"])


    print("running grid search")
    idx = 0
    for params in grid:    

        # random sampling...
        if random.random() > gridSampleSize:  # samples the grid
            continue # skips this set of parameters
        
        # check to see if these parameters have already been tried
        try:
            df = pd.read_excel(output_filename)
            if (df[param_columns] == params).all(1).any():
                continue
        except: pass

        # otherwise, go on as normal:
        
        # make the agent and memory buffer using the parameters
        memoryBuffer = MemoryBuffer(params['memory_buffer'])
        agent = Agent(num_moves, params['eps_start'], params['eps_end'], params['eps_decay'], memoryBuffer, params['batch_size'], params['learning_rate'], params['ams_grad'], params['gamma'], params['targetnet_update_rate'])

        # for the results of this episode (and set of parameters)
        idx += 1
        print(idx)
        RewardsList = []
        tempReward = []
        time_to_convergence = None
        best_average = 0
        best_episode = None


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
                
                if terminal:
                    next_state = None
                    tempReward.append(reward.item())   
                                

                # add trajectory to memory buffer and move to the next state
                agent.memory.push(state, action, next_state, reward)
                state = next_state

                # optimise the DQN model
                agent.optimize_model()

            # testing of the agent between 10 episode blocks
            if episode % 10 == 0 and episode > 0:
                # store the rewards of the last 10 training episode (NO SEPERATE TESTING HERE, SAVES SOME RUNNING TIME)
                RewardsList.append(sum(tempReward)/len(tempReward))
                tempReward = []

                # find the best average over 100 episodes
                running_avg = sum(RewardsList[-10:]) / 10   # each element in RewardsList is an average of 10 episodes
                if running_avg > best_average:
                    best_average = running_avg
                    best_episode = episode
                

                # early stopping
                # if the average of the previous 100 episodes was 1, we've hit convergence so stop (to try and save time)
                if running_avg == 1:
                    time_to_convergence = episode
                #    break


        # store the results in a dataframe, making a new row for this trial here
        tempDict = {"DQN_model" : DQN_model,
                    "avgRewards" : RewardsList,
                    "average_last_100_episodes" : running_avg,
                    "best_average_100_episodes" : best_average,
                    "time_of_peak" : best_episode,
                    "time_to_convergence" : time_to_convergence}
        
        resultsDict = {**params.copy(), **tempDict}  # make a line for in the results dict


        # Let parallel runs write to the same results file
        try:
            RESULTS_DATAFRAME = pd.read_excel(output_filename)
            RESULTS_DATAFRAME.loc[len(RESULTS_DATAFRAME)+1] = resultsDict
        except:
            RESULTS_DATAFRAME.loc[0] = resultsDict
            #output_filename = 'grid_backup.xlsx'
        RESULTS_DATAFRAME.drop(RESULTS_DATAFRAME.filter(regex="Unname"), axis=1, inplace=True)
        RESULTS_DATAFRAME.to_excel(output_filename) # saves on every iteration (in case this takes long, or crashes, we can still pull the results out)



if __name__ == '__main__':
    args = get_args_parser().parse_args()
    grid_search(args)

