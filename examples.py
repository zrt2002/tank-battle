import collections as cl
import numpy as np
from tankbattle.env.engine import TankBattle
from tankbattle.env.state import State, get_reward
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from tankbattle.env.model import DQN, ReplayBuffer


def train_DQN(agent: DQN,
              game: TankBattle,
              num_episodes,
              replay_buffer: ReplayBuffer,
              minimal_size,
              batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
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
                    episode_return += reward
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
                game.reset()
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list

def machine_control(two_players=False):
    replay_buffer = ReplayBuffer(5000)
    game = TankBattle(render=True, 
                      player1_human_control=False,
                      player2_human_control=False, 
                      two_players=two_players,
                      speed=1000, 
                      debug=False, 
                      frame_skip=5)
    num_of_actions = game.get_num_of_actions()

    lr = 1e-2
    num_episodes = 200
    action_dim = 5
    gamma = 0.98
    epsilon = 0.01
    target_update = 50
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = DQN(action_dim, lr, gamma, epsilon,
        target_update, device, 'DoubleDQN')
    train_DQN(agent, game, num_episodes, replay_buffer, minimal_size, batch_size)

    
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

    machine_control()
    # human_control()