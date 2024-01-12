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
        self.bullets_player = self.bullets_player
        self.bullets_enemy = self.bullets_enemy
        self.min_norm = min(abs(self.player_x - self.enemy_x), abs(self.player_y - self.enemy_y))
        self.max_norm = max(abs(self.player_x - self.enemy_x), abs(self.player_y - self.enemy_y))

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
    reward = naive_reward
    if next_state.min_norm < state.min_norm:
        reward += 0.05
    if next_state.max_norm < 2:
        reward -= 0.03 
    if state.aiming:
        reward += 1
        if action == GlobalConstants.FIRE_ACTION:
            reward += 1
    if is_terminal:
        reward -= 100
    return reward
    