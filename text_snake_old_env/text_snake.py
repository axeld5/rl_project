import gymnasium as gym
import numpy as np
from tqdm import tqdm

from text_snake_logic import *
from collections import namedtuple, deque
import random

import os, sys
import time

from model_DQN import *


class Direction():
    def __init__(self):
        self.RIGHT = 1
        self.LEFT = 2
        self.UP = 3
        self.DOWN = 4

direction = Direction()
Point = namedtuple('Point', 'x, y') # tuple which entries are named x and y


class TextSnakeEnv(gym.Env):
    def __init__(self,
                 screen_size,
                 LR=0.001,
                 MAX_MEMORY=100000,
                 BATCH_SIZE = 100,
                 GAMMA=1):
    
        self._screen_size = screen_size
        self.action_space = gym.spaces.Discrete(3)
        self.action_space_lut = {0: 'Left', 1: 'Straight', 2: 'Right'}

        self.observation_space = gym.spaces.MultiDiscrete(11) # 11 binary observations
        self._game = None

        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA)
        self.memory = deque(maxlen=MAX_MEMORY)

        self.BATCH_SIZE = BATCH_SIZE

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
        while True:
            if (dir_r and self._game.is_collision(point_r)) or \
                (dir_l and self._game.is_collision(point_l)) or \
                (dir_u and self._game.is_collision(point_u)) or \
                (dir_d and self._game.is_collision(point_d)):
                break
            else:
                dist_danger_s += 1
                point_l = Point(head.x - 1 - dist_danger_s, head.y)
                point_r = Point(head.x + 1 + dist_danger_s, head.y)
                point_u = Point(head.x, head.y - 1 - dist_danger_s)
                point_d = Point(head.x, head.y + 1 + dist_danger_s)
                
        
        # reset points
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        # check danger left
        dist_danger_l = 0
        while True:
            if (dir_d and self._game.is_collision(point_r)) or \
            (dir_u and self._game.is_collision(point_l)) or \
            (dir_r and self._game.is_collision(point_u)) or \
            (dir_l and self._game.is_collision(point_d)):
                break
            else:
                dist_danger_l += 1
                point_l = Point(head.x - 1 - dist_danger_l, head.y)
                point_r = Point(head.x + 1 + dist_danger_l, head.y)
                point_u = Point(head.x, head.y - 1 - dist_danger_l)
                point_d = Point(head.x, head.y + 1 + dist_danger_l)
                

        # reset points
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        # check danger right
        dist_danger_r = 0
        while True:
            if (dir_u and self._game.is_collision(point_r)) or \
            (dir_d and self._game.is_collision(point_l)) or \
            (dir_l and self._game.is_collision(point_u)) or \
            (dir_r and self._game.is_collision(point_d)):
                break
            else:
                dist_danger_r += 1
                point_l = Point(head.x - 1 - dist_danger_r, head.y)
                point_r = Point(head.x + 1 + dist_danger_r, head.y)
                point_u = Point(head.x, head.y - 1 - dist_danger_r)
                point_d = Point(head.x, head.y + 1 + dist_danger_r)
        if dist_danger_l > 0:
            dist_danger_l = 1 
        if dist_danger_r > 0:
            dist_danger_r = 1 
        if dist_danger_s > 0:
            dist_danger_s = 1
                

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

        return np.array(state, dtype=int)
    
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

        return obs, reward, done, info

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._game = SnakeLogic(self._screen_size)
        obs = self._get_observation()
        info = self._get_info()
        return obs, None, None, info


    def render(self, prev_r=None, prev_tail=None):
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

        if prev_r is None:
            r = np.zeros((self._screen_size[0]+1,self._screen_size[1]+1), dtype='int')
            r[0][:] = 4
            r[-1][:] = 5
            r[:,0] = 6
            r[:,-1] = 7
        else:
            r = prev_r

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
        
        return r_str, r_unflipped, point_tail

    def close(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, done, prediction_model):
        self.trainer.train_step(state, action, reward, next_state, done, prediction_model)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self, prediction_model):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, prediction_model)

    def choose_action(self, state, epsilon):
        random_int = random.uniform(0,1)
        if random_int > epsilon:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            action = self.argmax(prediction).item()
        else:
            action = self.action_space.sample()
        return action
    
    def argmax(self, prediction):
        """argmax with random tie-breaking
        """
        ties = np.where(prediction == torch.max(prediction))[0]
        return np.random.choice(ties)


        
    def train(self, epsilon_start, n_episodes, clone_model_freq = 10, eps_decay=0.9, eps_min=0.05):
        score_list = []
        epsilon = epsilon_start

        for ep in tqdm(range(n_episodes)):

            if ep % clone_model_freq == 0:
                prediction_model = copy.deepcopy(self.trainer.model)

            state = self.reset()[0]
            done = False
            epsilon = max(epsilon*eps_decay, eps_min)
            while True:
            # get old state
                state_old = self._get_observation()

                # get move
                action = self.choose_action(state_old, epsilon)

                # perform move and get new state
                state_new, reward, done, info = self.step(action)
                
                # train short memory
                self.train_short_memory(state_old, action, reward, state_new, done, prediction_model)

                # remember
                self.remember(state_old, action, reward, state_new, done)

                if done:
                    # train long memory, plot result
                    self.train_long_memory(prediction_model)

                    score_list.append(info["score"])
                    break
        return score_list # the trainer is updated
    
if __name__ == "__main__":
    env = TextSnakeEnv(screen_size = (15, 10))
    print("Training...")
    score_list = env.train(1, 1000)

    # once it is trained, print one trajectory using learnt policy

    # initiate environment
    obs, _, _, info = env.reset()

    # iterate
    r = None
    prev_tail = None
    while True:

        # Select next action
        prediction = env.trainer.model(torch.tensor(obs, dtype=torch.float))
        action = env.argmax(prediction).item()

        # Appy action and return new observation of the environment
        obs, reward, done, info = env.step(action)

        print(done)

        # Render the game
        os.system("cls")
        r_str, r, prev_tail = env.render(r, prev_tail)

        sys.stdout.write(r_str)
        time.sleep(0.2) # FPS

        # If player is dead break
        if done:
            break

    env.close()
