import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.error import DependencyNotInstalled

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

EMPTY = 0
COIN_1 = 1
COIN_2 = 2
AGENT_1 = 3
AGENT_2 = 4
BOTH = 5
BOTH_AND_COIN = 6

BLOCKSIZE = 20

class CoinGame(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size_h, size_v, default=True, positions=None):
        if size_h < 2 or size_v < 1:
            raise Exception("Invalid world size")
        
        self.rows = size_v
        self.cols = size_h
        self.default = default
        self.grid = np.zeros((self.rows, self.cols))
        self.curr_coin = COIN_1 if np.random.binomial(1, 0.5) else COIN_2

        self.init_pos = self.init_grid(positions)
        self.coin, self.a_1, self.a_2 = self.init_pos

        self.screen_width = size_h * BLOCKSIZE
        self.screen_height = size_v * BLOCKSIZE

    def init_grid(self, positions=None):
        if positions:
            pass
        elif self.rows==1 and self.cols==2:
            positions = np.array([0, 1, 1])
        elif self.default and self.rows==3 and self.cols==3:
            positions = np.array([4, 0, 2])
        else:
            positions = np.random.choice(self.rows*self.cols, 3, replace=False)

        coin, a_1, a_2 = positions
        if a_1==a_2:
            self.grid[coin//self.cols , coin%self.cols] = self.curr_coin
            self.grid[a_1//self.cols , a_1%self.cols] = BOTH
        else:
            self.grid[coin//self.cols , coin%self.cols] = self.curr_coin
            self.grid[a_1//self.cols , a_1%self.cols] = AGENT_1
            self.grid[a_2//self.cols , a_2%self.cols] = AGENT_2

        return positions

    def step(self, actions):
        a1, a2 = actions
        a_1_x, a_1_y = self.a_1//self.cols, self.a_1%self.cols
        a_2_x, a_2_y = self.a_2//self.cols, self.a_2%self.cols

        self.grid[a_1_y, a_1_x] = EMPTY
        self.grid[a_2_y, a_2_x] = EMPTY

        new_a_1_y, new_a_1_x = self.calculate_pos(a_1_x, a_1_y, a1)
        new_a_2_y, new_a_2_x = self.calculate_pos(a_2_x, a_2_y, a2)

        self.a_1 = new_a_1_y*self.cols + new_a_1_x
        self.a_2 = new_a_2_y*self.cols + new_a_2_x

        r1, r2 = 0, 0

        if self.a_1==self.coin or self.a_2==self.coin:
            terminated = True
            # A1 has coin 1
            if self.a_1==self.coin and self.curr_coin==COIN_1:
                r1 = 1
            # A2 has coin 2
            if self.a_2==self.coin and self.curr_coin==COIN_2:
                r1 = 1
            # A1 has coin 2
            if self.a_1==self.coin and self.curr_coin==COIN_2:
                r1, r2 = 1, -1
            # A2 has coin 1
            if self.a_2==self.coin and self.curr_coin==COIN_1:
                r1, r2 = -1, 1

        reward = np.array([r1, r2])

        if self.a_1==self.coin and self.a_2==self.coin:
            self.grid[new_a_1_y, new_a_1_x] = BOTH_AND_COIN
        if self.a_1==self.a_2:
            self.grid[new_a_1_y, new_a_1_x] = BOTH
        else:
            self.grid[new_a_1_y, new_a_1_x] = AGENT_1
            self.grid[new_a_2_y, new_a_2_x] = AGENT_2

        return self.grid, reward, terminated, False, {}

    def calculate_pos(self, a_x, a_y, a):
        if a==UP:
            a_y = max(0, a_y-1)
        elif a==DOWN:
            a_y = min(0, a_y+1)
        elif a==LEFT:
            a_x = min(0, a_x+1)
        elif a==RIGHT:
            a_x = min(0, a_x+1)
        return (a_y, a_x)


    def reset(self):
        self.coin, self.a_1, self.a_2 = self.init_pos
        self.init_grid(self.init_pos)

        if self.render_mode == "human":
            self.render()

        return self.grid, {}

    def render(self, mode='human', close=False):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((0, 0, 0))

        for x in range(0, self.screen_width, BLOCKSIZE):
            for y in range(0, self.screen_height, BLOCKSIZE):
                rect = pygame.Rect(x, y, BLOCKSIZE, BLOCKSIZE)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)
        
