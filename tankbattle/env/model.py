import random
import time
now = time.strftime(
    '%Y-%m-%d-%H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000)
)

import numpy as np
import torch
import torch.nn.functional as F

import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tankbattle.env.engine import TankBattle
from tankbattle.env.state import State, get_reward

writer = SummaryWriter()
class Qnet(torch.nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, kernel_size=4, padding=2)
        self.pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(64, 48)
        self.linear2 = torch.nn.Linear(48, 32)
        self.linear3 = torch.nn.Linear(32, 16)
        self.linear4 = torch.nn.Linear(16, 5)
        self.activation = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pooling(x)
        x = self.activation(self.conv2(x))
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        return self.linear4(x)    

class DQN:
    def __init__(self,
                 action_dim = 5,
                 learning_rate = 1e-2,
                 gamma = 0.98,
                 epsilon = 0.1,
                 target_update = 50,
                 device = torch.device('cpu'),
                 dqn_type='DoubleDQN'):
        self.action_dim = action_dim
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
            state = torch.tensor(state, dtype=torch.float).reshape(1, 4, 11, 11).to(self.device)
            action = self.q_net(state).argmax().item()
        self.epsilon = (self.epsilon - 0.1) / (1 + 0.00001) + 0.1
        return action
    def max_q_value(self, state):
        state = torch.tensor(state, dtype=torch.float).reshape(1, 4, 11, 11).to(self.device)
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
    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        
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
def train_DQN(
    agent: DQN,
    game: TankBattle,
    n_epochs,
    replay_buffer: ReplayBuffer,
    minimal_size,
    batch_size,
    kill_multiple,
):
    max_q_value_list = []
    max_q_value = 0
    for epoch in tqdm(range(n_epochs)):
        epoch_return = 0
        epoch_true_score = 0
        state = State(game)
        is_terminal = False
        while not is_terminal:
            action = agent.take_action(state.board)
            max_q_value = (
                agent.max_q_value(state.board) * 0.005 + max_q_value * 0.995
            )
            max_q_value_list.append(max_q_value)
            naive_reward = game.step(action)[0]
            next_state = State(game)
            is_terminal = game.is_terminal()
            reward = get_reward(
                state, action, next_state, naive_reward, is_terminal, kill_multiple
            )
            replay_buffer.add(
                state.board, action, reward, next_state.board, is_terminal
            )
            state = next_state
            epoch_return += reward
            epoch_true_score += naive_reward
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d,
                }
                agent.update(transition_dict)
            writer.add_scalar('max_q_value', max_q_value, epoch)
        game.reset()
        writer.add_scalar('epoch_return', epoch_return, epoch)
        writer.add_scalar('epoch_true_score', epoch_true_score, epoch)
        writer.add_scalar('epsilon', agent.epsilon, epoch)
        if epoch % 20 == 0:
            agent.save(f'ckpts/{now}/model_{epoch}.pth')