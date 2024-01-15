import multiprocessing
import os

import numpy as np
import pandas as pd
import torch

from tankbattle.env.constants import GlobalConstants
from tankbattle.env.engine import TankBattle
from tankbattle.env.model import DQN
from tankbattle.env.state import State

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


def evaluate_one_model(
    model_path: str,
    random_fire_rate: float = None,
    n_epochs: int = 500,
    epsilon: float = 0.1,
    render: bool = False,
):
    game = TankBattle(
        render=render,
        player1_human_control=False,
        player2_human_control=False,
        two_players=False,
        speed=60,
        debug=False,
        frame_skip=5,
    )
    if model_path:
        print(f'start: {model_path}, epsilon: {epsilon}')
        lr = 1e-2
        action_dim = 5
        gamma = 0.98
        target_update = 50
        device = torch.device('cpu')
        agent = DQN(action_dim, lr, gamma, epsilon, target_update, device, 'DoubleDQN')
        agent.load(model_path)
        agent.epsilon = epsilon
        epoch_true_scores = []
        epoch_true_scores.append(model_path)
        for epoch in range(n_epochs):
            print(f'epoch {epoch}')
            epoch_true_score = 0
            state = State(game)
            is_terminal = False
            while not is_terminal:
                action = agent.take_action(state.board)
                naive_reward = game.step(action)[0]
                is_terminal = game.is_terminal()
                epoch_true_score += naive_reward
            game.reset()
            epoch_true_scores.append(epoch_true_score)
    elif random_fire_rate:
        print(f'random_fire_rate: {random_fire_rate}')
        epoch_true_scores = []
        epoch_true_scores.append(random_fire_rate)
        for epoch in range(n_epochs):
            print(f'epoch {epoch}')
            epoch_true_score = 0
            is_terminal = False
            while not is_terminal:
                if np.random.random() < random_fire_rate:
                    action = GlobalConstants.FIRE_ACTION
                else:
                    action = np.random.randint(4)
                naive_reward = game.step(action)[0]
                is_terminal = game.is_terminal()
                epoch_true_score += naive_reward
            game.reset()
            epoch_true_scores.append(epoch_true_score)
    print(epoch_true_scores)
    average = sum(epoch_true_scores[1:]) / len(epoch_true_scores[1:])
    epoch_true_scores.append(average)
    return epoch_true_scores

def evaluate_many_models(
    model_paths: list[str],
    random_fire_rates: list[float] = None,
    processes: int = 12,
    eval_epochs: int = 500,
    epsilon: float = 0.1,
    render: bool = False,
    csv_save_path: str = None,
):
    pool = multiprocessing.Pool(processes)
    if model_paths:
        results = [
            pool.apply_async(
                evaluate_one_model, args=(model_path, None, eval_epochs, epsilon)
            )
            for model_path in model_paths
        ]
    elif random_fire_rates:
        results = [
            pool.apply_async(
                evaluate_one_model,
                args=(None, random_fire_rate, eval_epochs, epsilon, render),
            )
            for random_fire_rate in random_fire_rates
        ]
    result_list = []
    for res in results:
        result_list.append(res.get())
    df = pd.DataFrame(result_list)
    df.to_csv(csv_save_path, index=False)
    pool.close()


if __name__ == '__main__':
    model_paths = None
    random_fire_rates = None
    # model_paths = [f'ckpts/2024-01-12-21-01-43/model_{i}.pth' for i in range(0, 500, 20)]
    model_paths = ['ckpts/2024-01-13-00-37-51/model_1200.pth']
    # print(model_paths)
    # random_fire_rates = np.linspace(0.1, 1, 14).tolist()
    print(random_fire_rates)
    evaluate_many_models(
        model_paths=model_paths,
        processes=1,
        eval_epochs=500,
        epsilon=0.1,
        random_fire_rates=random_fire_rates,
        render=False,
        csv_save_path='DQN_1200.csv',
    )
