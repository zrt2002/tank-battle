import collections as cl
import numpy as np
from tankbattle.env.engine import TankBattle
from tankbattle.env.state import State, get_reward
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torch import nn
from tankbattle.env.model import DQN, ReplayBuffer
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import os
from typing import Union

now = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(int(round(time.time()*1000))/1000))
writer = SummaryWriter()

def train_DQN(agent: DQN,
              game: TankBattle,
              n_epochs,
              replay_buffer: ReplayBuffer,
              minimal_size,
              batch_size):
    max_q_value_list = []
    max_q_value = 0
    for epoch in tqdm(range(n_epochs)):
        epoch_return = 0
        epoch_true_score = 0
        state = State(game)
        is_terminal = False
        while not is_terminal:
            action = agent.take_action(state.board)
            max_q_value = agent.max_q_value(state.board) * 0.005 + max_q_value * 0.995  # 平滑处理
            max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值                    
            naive_reward = game.step(action)[0]
            next_state = State(game)
            is_terminal = game.is_terminal()
            reward = get_reward(state, action, next_state, naive_reward, is_terminal)
            replay_buffer.add(state.board, action, reward, next_state.board, is_terminal)
            state = next_state
            epoch_return += reward
            epoch_true_score += naive_reward
            print(f'local reward: {epoch_return}')
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                    batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                agent.update(transition_dict)
            writer.add_scalar("max_q_value", max_q_value, epoch)
        game.reset()
        print(f'epoch {epoch}: reward {epoch_return}, true_reward {epoch_true_score}')
        writer.add_scalar("epoch_return", epoch_return, epoch)
        writer.add_scalar("epoch_true_score", epoch_true_score, epoch)
        writer.add_scalar("epsilon", agent.epsilon, epoch)
        if epoch % 20 == 0:
            agent.save(f'ckpts/{now}/model_{epoch}.pth')
def monkey_try(game: TankBattle, n_epochs):
    for epoch in tqdm(range(n_epochs)):
        epoch_true_score = 0
        is_terminal = False
        while not is_terminal:
            action = random.randint(0, 4)
            naive_reward = game.step(action)[0]
            is_terminal = game.is_terminal()
            epoch_true_score += naive_reward
        game.reset()
        writer.add_scalar("epoch_true_score", epoch_true_score, epoch)

def evaluate_DQN(agent: DQN,
              game: TankBattle,
              n_epochs,
            #   replay_buffer: ReplayBuffer,
            #   minimal_size,
            #   batch_size
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
            max_q_value = agent.max_q_value(state.board) * 0.005 + max_q_value * 0.995  # 平滑处理
            max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值                    
            naive_reward = game.step(action)[0]
            next_state = State(game)
            is_terminal = game.is_terminal()
            reward = get_reward(state, action, next_state, naive_reward, is_terminal)
            # replay_buffer.add(state.board, action, reward, next_state.board, is_terminal)
            state = next_state
            epoch_return += reward
            epoch_true_score += naive_reward
            print(f'local reward: {epoch_return}')
            # if replay_buffer.size() > minimal_size:
                # b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                    # batch_size)
                # transition_dict = {
                #     'states': b_s,
                #     'actions': b_a,
                #     'next_states': b_ns,
                #     'rewards': b_r,
                #     'dones': b_d
                # }
                # agent.update(transition_dict)
            writer.add_scalar("max_q_value", max_q_value, epoch)
        game.reset()
        print(f'epoch {epoch}: reward {epoch_return}, true_reward {epoch_true_score}')
        writer.add_scalar("epoch_return", epoch_return, epoch)
        writer.add_scalar("epoch_true_score", epoch_true_score, epoch)
        writer.add_scalar("epsilon", agent.epsilon, epoch)
        # if epoch % 20 == 0:
            # agent.save(f'ckpts/{now}/model_{epoch}.pth')

def machine_control(two_players=False,
                    ckpt: Union[str, None] = None,
                    train: bool = False,
                    monkey: bool = False):
    replay_buffer = ReplayBuffer(5000)
    game = TankBattle(render=True, 
                      player1_human_control=False,
                      player2_human_control=False, 
                      two_players=two_players,
                      speed=20000, 
                      debug=False, 
                      frame_skip=5)
    num_of_actions = game.get_num_of_actions()

    lr = 1e-2
    n_epochs = 500
    action_dim = 5
    gamma = 0.98
    epsilon = 0.5
    target_update = 50
    buffer_size = 5000
    minimal_size = 2000 #TODO
    batch_size = 64
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    os.makedirs(f'ckpts/{now}', exist_ok=True)
    if monkey:
        monkey_try(game, n_epochs)
        return 0
    if ckpt:
        agent = DQN(action_dim, lr, gamma, 0.1,
        target_update, device, 'DoubleDQN')
        agent.load(ckpt)
        if train:
            train_DQN(agent, game, n_epochs, replay_buffer, minimal_size, batch_size)
        else:
            evaluate_DQN(agent, game, n_epochs)
    else:
        agent = DQN(action_dim, lr, gamma, epsilon,
        target_update, device, 'DoubleDQN')


    # episodes_list = list(range(len(return_list)))
    # mv_return = rl_utils.moving_average(return_list, 5)
    # plt.plot(return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.savefig('returns.png')
    # plt.cla()
    # frames_list = list(range(len(max_q_value_list)))
    # plt.plot(frames_list, max_q_value_list)
    # plt.axhline(0, c='orange', ls='--')
    # plt.axhline(10, c='red', ls='--')
    # plt.xlabel('Frames')
    # plt.ylabel('Q value')
    # plt.savefig('q_values.png')

def human_control(two_players=False):

    game = TankBattle(render=True, player1_human_control=True, player2_human_control=True, two_players=two_players,
                      speed=60, debug=True, frame_skip=5)

    print("Press 'Space' to fire and arrow keys to control the tank !")

    game.reset()
    scores = []

    for step in range(100000):
        game.render()

        terminal = game.is_terminal()
        if terminal:
            print("P1 Score:", game.total_score_p1)
            if two_players:
                print("P2 Score:", game.total_score_p2)
            print("Total Score", game.total_score)
            print("Current steps:", step)
            scores.append(game.total_score)
            game.reset()

    print(scores)


if __name__ == '__main__':

    machine_control(two_players=False, ckpt=r'C:\Users\rinto\Documents\GitHub\tank-battle\ckpts\2024-01-12-21-01-43\model_240.pth', train=False, monkey=True)
    # human_control()