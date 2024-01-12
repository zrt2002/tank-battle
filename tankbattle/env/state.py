import numpy as np
from tankbattle.env.constants import GlobalConstants
from tankbattle.env.engine import TankBattle
from tankbattle.env.sprites.tank import TankSprite
from tankbattle.env.sprites.bullet import BulletSprite

class State:
    # @staticmethod
    def __init__(self, game: TankBattle):
        '''
        Remark: 
        1. game.enemies, game.bullets_player and game.bullets_enemy are all pygame.sprite.Group. They can only be accessed by for-loop and can't be accessed by index.
        2. the 0th row and 0th column of the map are walls, so we subtract 1 from all coordinates so as to reduce redudant dimension.
        '''
        self.current_board = np.zeros([4, 11, 11])
        self.player_x = game.player1.pos_x - 1
        self.player_y = game.player1.pos_y - 1
        self.player_dir = game.player1.direction
        self.current_board[0, self.player_x, self.player_y] = self.player_dir
        enemy: TankSprite
        for enemy in game.enemies:
            self.enemy_x = enemy.pos_x - 1
            self.enemy_y = enemy.pos_y - 1 
            self.enemy_dir = enemy.direction
            self.current_board[1, self.enemy_x, self.enemy_y] = self.enemy_dir

        bullet: BulletSprite
        self.bullets_player: list = []

        for bullet in game.bullets_player:
            self.bullet_x = bullet.pos_x - 1
            self.bullet_y = bullet.pos_y - 1
            self.bullet_dir = bullet.direction
            self.current_board[2, self.bullet_x, self.bullet_y] = self.bullet_dir
            self.bullets_player.append([self.bullet_x, self.bullet_y, self.bullet_dir])
        
        self.bullets_enemy: list = []
        for bullet in game.bullets_enemy:
            self.bullet_x = bullet.pos_x - 1
            self.bullet_y = bullet.pos_y - 1
            self.bullet_dir = bullet.direction
            self.current_board[3, self.bullet_x, self.bullet_y] = self.bullet_dir
            self.bullets_enemy.append([self.bullet_x, self.bullet_y, self.bullet_dir])

        self.board = self.current_board
        self.min_norm = min(abs(self.player_x - self.enemy_x), abs(self.player_y - self.enemy_y))
        self.max_norm = max(abs(self.player_x - self.enemy_x), abs(self.player_y - self.enemy_y))

        self.enemy_age = game.enemy_age
        self.aiming = self.is_aiming()

    def is_aiming(self):
        if self.player_x == self.enemy_x:
            if (self.player_y > self.enemy_y) and self.player_dir == GlobalConstants.UP_ACTION:
                return True
            elif (self.player_y < self.enemy_y) and self.player_dir == GlobalConstants.DOWN_ACTION:
                return True
        elif self.player_y == self.enemy_y:
            if (self.player_x > self.enemy_x) and self.player_dir == GlobalConstants.LEFT_ACTION:
                return True
            elif (self.player_x < self.enemy_x) and self.player_dir == GlobalConstants.RIGHT_ACTION:
                return True
        return False


def get_reward(state: State, action, next_state: State, naive_reward, is_terminal):
    reward = naive_reward * 20
    # reward -= 1.01 ** (state.enemy_age / 1000) / 100
    # if next_state.min_norm < state.min_norm:
        # reward += 0.1
    if next_state.max_norm < 2:
        reward -= 0.1 
    if state.aiming:
        reward += 0.114514223
        if action == GlobalConstants.FIRE_ACTION:
            reward += 5
    # avoid fatal bullets
    dangerous_grids = []
    for bullet in state.bullets_enemy:
        bullet_x = bullet[0]
        bullet_y = bullet[1]
        bullet_dir = bullet[2]
        if bullet_dir == GlobalConstants.UP_ACTION:
            dangerous_grids.append([bullet_x, bullet_y - 1])
            dangerous_grids.append([bullet_x, bullet_y - 2])
            dangerous_grids.append([bullet_x, bullet_y - 3])
            dangerous_grids.append([bullet_x, bullet_y - 4])
        elif bullet_dir == GlobalConstants.DOWN_ACTION:
            dangerous_grids.append([bullet_x, bullet_y + 1])
            dangerous_grids.append([bullet_x, bullet_y + 2])
            dangerous_grids.append([bullet_x, bullet_y + 3])
            dangerous_grids.append([bullet_x, bullet_y + 4])
        elif bullet_dir == GlobalConstants.LEFT_ACTION:
            dangerous_grids.append([bullet_x - 1, bullet_y])
            dangerous_grids.append([bullet_x - 2, bullet_y])
            dangerous_grids.append([bullet_x - 3, bullet_y])
            dangerous_grids.append([bullet_x - 4, bullet_y])
        elif bullet_dir == GlobalConstants.RIGHT_ACTION:
            dangerous_grids.append([bullet_x + 1, bullet_y])
            dangerous_grids.append([bullet_x + 2, bullet_y])
            dangerous_grids.append([bullet_x + 3, bullet_y])
            dangerous_grids.append([bullet_x + 4, bullet_y])
    dangerous_actions = []
    if [state.player_x, state.player_y - 1] in dangerous_grids:
        dangerous_actions.append(GlobalConstants.UP_ACTION)
    if [state.player_x, state.player_y + 1] in dangerous_grids:
        dangerous_actions.append(GlobalConstants.DOWN_ACTION)
    if [state.player_x - 1, state.player_y] in dangerous_grids:
        dangerous_actions.append(GlobalConstants.LEFT_ACTION)
    if [state.player_x + 1, state.player_y] in dangerous_grids:
        dangerous_actions.append(GlobalConstants.RIGHT_ACTION)
    if [state.player_x, state.player_y] in dangerous_grids:
        dangerous_actions.append(GlobalConstants.FIRE_ACTION)
    if (dangerous_actions != []) and (action not in dangerous_actions):
        reward += 10

    # moving into walls
    if ((state.player_x == 0 and action == GlobalConstants.LEFT_ACTION)
        or (state.player_x == 10 and action == GlobalConstants.RIGHT_ACTION)
        or (state.player_y == 0 and action == GlobalConstants.UP_ACTION)
        or (state.player_y == 10 and action == GlobalConstants.DOWN_ACTION)
        ):
        reward -= 2
    # shooting into walls
    if ( (state.player_x == 0 and state.player_dir == GlobalConstants.LEFT_ACTION)
          or (state.player_x == 10 and state.player_dir == GlobalConstants.RIGHT_ACTION)
          or (state.player_y == 0 and state.player_dir == GlobalConstants.UP_ACTION)
          or (state.player_y == 10 and state.player_dir == GlobalConstants.DOWN_ACTION)
          ) and (action == GlobalConstants.FIRE_ACTION):
        reward -= 2
    if is_terminal:
        reward -= 100
        print(f"terminal, enemy_age = {state.enemy_age}")
    return reward
    