import argparse

from typing import Union


from tankbattle.env.engine import TankBattle
from tankbattle.env.model import DQN
from tankbattle.env.state import State




def play_game(
    model_path: Union[str, None] = None,
):
    if model_path:
        player1_human_control = False
    else:
        player1_human_control = True
    print(player1_human_control)
    game = TankBattle(
        render=True,
        player1_human_control=player1_human_control,
        player2_human_control=True,
        two_players=False,
        speed=120,
        debug=False,
        frame_skip=5,
    )
    if model_path:
        agent = DQN()
        agent.load(model_path)
        for epoch in range(1000):
            print(f'epoch {epoch}')
            epoch_true_score = 0
            state = State(game)
            is_terminal = False
            while not is_terminal:
                action = agent.take_action(state.board)
                naive_reward = game.step(action)[0]
                is_terminal = game.is_terminal()
                epoch_true_score += naive_reward
            print(epoch_true_score)
            game.reset()
    else:
        print('Press J to fire and WASD to control the tank!')
        game.reset()
        scores = []
        with open('./log.txt', 'w') as f:
            for step in range(100000):
                game.render()
                terminal = game.is_terminal()
                if terminal:
                    scores.append(game.total_score)
                    f.write(str(game.total_score) + '\n')
                    game.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=None)
    model_path: str = parser.parse_args().model_path
    play_game(model_path=model_path)