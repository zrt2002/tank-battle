import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from tankbattle.env.state import State, get_reward

class Qnet(torch.nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, kernel_size=4, padding=2)
        self.pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(64, 32)
        self.linear2 = torch.nn.Linear(32, 10)
        self.linear3 = torch.nn.Linear(10, 1)
        self.activation = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pooling(x)
        x = self.activation(self.conv2(x))
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        return self.linear3(x)    

    
class VAnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, kernel_size=4, padding=2)
        self.pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(64, 32)
        self.linear2 = torch.nn.Linear(32, 10)
        self.linear3 = torch.nn.Linear(10, 1)
        self.activation = torch.nn.Sigmoid()  # 共享网络部分
        self.fc_A = torch.nn.Linear(10, 5)
        self.fc_V = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pooling(x)
        x = self.activation(self.conv2(x))
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        A = self.fc_A(x)
        V = self.fc_V(x)
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q

class DQN:
    def __init__(self,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 device,
                 dqn_type='VanillaDQN'):
        self.action_dim = action_dim
        if dqn_type == 'DuelingDQN':  # Dueling DQN采取不一样的网络框架
            self.q_net = VAnet().to(device)
            self.target_q_net = VAnet().to(device)
        else:
            self.q_net = Qnet().to(device)
            self.target_q_net = Qnet().to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        if self.dqn_type == 'DoubleDQN':
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(
                1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
                -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

class ReplayBuffer(list):
    def __init__(self, buffer_size):
        super(ReplayBuffer, self).__init__()
        self.buffer_size = buffer_size
    def add(self, state, action, reward, next_state, is_terminal):
        self.append((state, action, reward, next_state, is_terminal))
        if len(self) > self.buffer_size:
            self.pop(0)
    def size(self):
        return len(self)
    def sample(self, batch_size):
        batch = random.sample(self, batch_size)
        states = [sample[0] for sample in batch]
        actions = [sample[1] for sample in batch]
        rewards = [sample[2] for sample in batch]
        next_states = [sample[3] for sample in batch]
        is_terminals = [sample[4] for sample in batch]
        return states, actions, rewards, next_states, is_terminals
