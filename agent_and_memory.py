from collections import namedtuple, deque
from skimage.transform import resize
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class MemoryBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) # iterable deque data structure

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    







class DQN(nn.Module):
    def __init__(self, number_of_actions):
        super(DQN, self).__init__()
        # conv layers
        self.conv1 = torch.nn.Conv2d(4, 32, 5, stride=3)   # modify input shape to match your input data size
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=2)
        # fully connected layers
        self.fc1 = torch.nn.Linear(576, 128)
        self.fc2 = torch.nn.Linear(128, number_of_actions)

        # mat1 and mat2 shapes cannot be multiplied (8x1024 and 16x512)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        #print('doing conv3')
        #x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
















#### AGENT CODE

class Agent(object):
    def __init__(self, num_moves, eps_start, eps_min, eps_decay, memory, batch_size, learning_rate, amsgrad, gamma, target_network_update_rate):
        self.gamma = gamma
        #self.optimizer = optimizer
        self.batch_size = batch_size
        self.memory = memory
        self.num_possible_moves = num_moves
        self.epsilon = eps_start
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.episode = 0
        self.steps = 0
        self.target_network_update_rate = target_network_update_rate

        self.model = DQN(self.num_possible_moves).to(device)
        self.target_network = DQN(self.num_possible_moves).to(device)
        self.target_network.load_state_dict(self.model.state_dict())


        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=amsgrad)

    
    def select_action(self, state, testing = False): 
        # the 'testing' variable is a way for us to enforce that the agent is exploiting (not exploring) during the 10 episodes of testing
        if np.random.rand() <= self.epsilon and testing == False:
            return torch.tensor([[random.randrange(self.num_possible_moves)]], device=device, dtype=torch.long)

        q_values = self.model(state)
        action = q_values.max(1)[1].view(1, 1)

        return action # returns a tensor of shape [[n]] (where n is the action number)

    def optimize_model(self):
        self.steps += 1
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # use of masking to handle the final states (where there is no next state)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        # concatenate the states, actions and rewards into batches
        batch_of_states = torch.cat(batch.state) 
        batch_of_actions = torch.cat(batch.action)
        batch_of_rewards = torch.cat(batch.reward)

        # get the Q(s_t, a) values for the current state and the chosen action
        state_action_values = self.model(batch_of_states).gather(1, batch_of_actions)

        # Compute state-action values for all next states using the target network:  max(Q(s_{t+1}, a)).
        # 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]  # get the max Q value
        
        # set the temporal difference learning target
        TD_targets = (batch_of_rewards + (self.gamma * next_state_values)  ).unsqueeze(1)


        # Compute the Huber loss
        criterion = nn.SmoothL1Loss()
        TD_loss = criterion(state_action_values, TD_targets)

        # Optimize the model
        self.optimizer.zero_grad()
        TD_loss.backward()
        
        # clip the losses (Huber loss)
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()


        # update the target network every certain number of steps
        if self.steps % self.target_network_update_rate == 0:
            self.overwrite_target_network()
            

        self.update_eps()

        del non_final_mask, non_final_next_states, batch_of_states, batch_of_actions, batch_of_rewards, state_action_values, next_state_values, TD_targets


    def update_eps(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.eps_start * np.exp(-self.episode/self.eps_decay)
            # keep the epsilon value from going below the minimum
            if self.epsilon < self.eps_min: 
                self.epsilon = self.eps_min

    # update the target network by overwriting it with the current model
    def overwrite_target_network(self):
        self.target_network.load_state_dict(self.model.state_dict())

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)