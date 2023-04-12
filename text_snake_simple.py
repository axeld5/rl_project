import gymnasium as gym
import numpy as np
import torch

from text_snake_logic import *
from collections import namedtuple

import os, sys
import time

from dqn_functions.dqn_trainer import QTrainer


class Direction():
    def __init__(self):
        self.RIGHT = 1
        self.LEFT = 2
        self.UP = 3
        self.DOWN = 4

direction = Direction()
Point = namedtuple('Point', 'x, y') # tuple which entries are named x and y


class TextSnakeEnvSimple(gym.Env):
    def __init__(self, screen_size):
    
        self._screen_size = screen_size
        self.action_space = gym.spaces.Discrete(3)
        self.action_space_lut = {0: 'Left', 1: 'Straight', 2: 'Right'}

        self.observation_space = gym.spaces.MultiBinary(11) # 11 binary observations
        self._game = None

        self.r_str = None 
        self.r_unflipped = None
        self.point_tail = None 


    def _get_observation(self):
        head = self._game.snake[0]

        # Boolean indicating snake's direction, from the point of vue of someone looking at the screen
        dir_l = self._game.direction == direction.LEFT
        dir_r = self._game.direction == direction.RIGHT
        dir_u = self._game.direction == direction.UP
        dir_d = self._game.direction == direction.DOWN

        # point left, right, up and down the head, from the point of vue of someone looking at the screen
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        # check danger straight
        dist_danger_s = 0
        if (dir_r and self._game.is_collision(point_r)) or \
                (dir_l and self._game.is_collision(point_l)) or \
                (dir_u and self._game.is_collision(point_u)) or \
                (dir_d and self._game.is_collision(point_d)):
            dist_danger_s = 1
        else:
            dist_danger_s = 0
        
        # reset points
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        # check danger left
        dist_danger_l = 0
        if (dir_d and self._game.is_collision(point_r)) or \
            (dir_u and self._game.is_collision(point_l)) or \
            (dir_r and self._game.is_collision(point_u)) or \
            (dir_l and self._game.is_collision(point_d)):
            dist_danger_l = 1
        else:
            dist_danger_l = 0 
                

        # reset points
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        # check danger right
        dist_danger_r = 0
        if (dir_u and self._game.is_collision(point_r)) or \
            (dir_d and self._game.is_collision(point_l)) or \
            (dir_l and self._game.is_collision(point_u)) or \
            (dir_r and self._game.is_collision(point_d)):
            dist_danger_r = 1
        else:
            dist_danger_r = 0
                

        state = [
            # Distance to danger left (snake's point of vue)
            dist_danger_l,

            # Distance to danger straight (snake's point of vue)
            dist_danger_s,

            # Distance to danger right (snake's point of vue)
            dist_danger_r,

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location (from the point of vue of someone seeing the screen)
            self._game.food.x < self._game.head.x,  # food left
            self._game.food.x > self._game.head.x,  # food right
            self._game.food.y < self._game.head.y,  # food up
            self._game.food.y > self._game.head.y  # food down
            ]

        return np.array(state, dtype=np.int8)
    
    def _get_info(self):
        obs = self._get_observation()
        return {
            "score": self._game.score,
            "head": self._game.head
        }
    
    def step(self, action):
        """
        Given an action it returns
        * current observations
        * a reward
        * a boolean : game over or not
        * an info dictionary with score and head coordinates
        """
        
        reward, done, score = self._game.update_state(action)
        obs = self._get_observation()

        info = self._get_info()

        return obs, reward, done, False, info

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._game = SnakeLogic(self._screen_size)
        self.r_unflipped = None
        self.point_tail = None 
        obs = self._get_observation()
        info = self._get_info()
        return obs, info


    def render(self):
        """
        0 : background (' ')
        1 : snake-lr ('-')
        2 : snake-up ('|')
        3 : dead snake ('*')
        4 : left wall ('[')
        5 : right wall (']')
        6 : up wall ('-')
        7 : down wall ('^')
        8 : apple ('@')
        9 : snake-head ('o')
        """
        lut = {0:' ', 
            1:gym.utils.colorize('-',"yellow"),
            2:gym.utils.colorize('|',"yellow"),
            3:gym.utils.colorize('*',"red"),
            4:'[',
            5:']',
            6:'-',
            7:'^',
            8:gym.utils.colorize('@',"green"),
            9:gym.utils.colorize('o',"blue")}

        if self.r_unflipped is None:
            r = np.zeros((self._screen_size[0]+1,self._screen_size[1]+1), dtype='int')
            r[0][:] = 4
            r[-1][:] = 5
            r[:,0] = 6
            r[:,-1] = 7
        else:
            r = self.r_unflipped

        point_tail = self._game.snake[-1]
        

        if not self._game.done:
            # print head as 'o'
            r[self._game.snake[0].x, self._game.snake[0].y] = 9
        else:
            # print head as '*'
            print("Done, printing *")
            r[self._game.snake[0].x, self._game.snake[0].y] = 3 

        # print food
        r[self._game.food.x, self._game.food.y] = 8
        
        # change direction of body before head
        if self._game.direction == direction.RIGHT or self._game.direction == direction.LEFT:
            r[self._game.snake[1].x, self._game.snake[1].y] = 1
        else:
            r[self._game.snake[1].x, self._game.snake[1].y] = 2
        
        prev_tail = self.point_tail
        # remove previous tail
        if prev_tail is not None:
            r[prev_tail.x, prev_tail.y] = 0
        # if not previous tail is given, that means that it is the initial observation.
        # you have to plot the first tail :
        else:
            if self._game.direction == direction.RIGHT or self._game.direction == direction.LEFT:
                r[self._game.snake[2].x, self._game.snake[2].y] = 1
            else:
                r[self._game.snake[2].x, self._game.snake[2].y] = 2
            

        self._render = r
        r_unflipped = r.copy()
        r = np.flipud(np.rot90(r,1))
        
        
        r_str = 'Text Snake!\nScore: {}\n'.format(self._game.score)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                r_str += lut[r[i,j]]
            r_str += '\n'
        if self._game.player_last_action is None:
            r_str += 'Game initialized!'
        else:
            r_str += 'Player Action ({})\n'.format(self.action_space_lut[self._game.player_last_action])
        self.r_str = r_str 
        self.r_unflipped = r_unflipped 
        self.point_tail = point_tail 

        os.system("cls")
        sys.stdout.write(self.r_str)
        time.sleep(0.2)
        #return r_str, r_unflipped, point_tail

    def close(self):
        pass
    
if __name__ == "__main__":
    env = TextSnakeEnvSimple(screen_size = (15, 10))
    print("Training...")
    qtrainer = QTrainer()
    num_runs = 1
    for run in range(num_runs):
        score_list = qtrainer.train(env, 1000)

    # once it is trained, print one trajectory using learnt policy

    # initiate environment
    obs, info = env.reset()

    # iterate
    r = None
    prev_tail = None
    while True:

        # Select next action
        prediction = qtrainer.model(torch.tensor(obs, dtype=torch.float))
        action = qtrainer.argmax(prediction).item()

        # Appy action and return new observation of the environment
        obs, reward, done, _, info = env.step(action)

        print(done)

        # Render the game
        env.render() # FPS

        # If player is dead break
        if done:
            break

    env.close()
