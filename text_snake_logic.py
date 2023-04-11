import random
from gymnasium import logger
import numpy as np

from collections import namedtuple


class Direction():
    def __init__(self):
        self.RIGHT = 1
        self.LEFT = 2
        self.UP = 3
        self.DOWN = 4

direction = Direction()
Point = namedtuple('Point', 'x, y') # tuple which entries are named x and y


class SnakeLogic:
    def __init__(self, screen_size):
        self._screen_width = screen_size[0]
        self._screen_height = screen_size[1]

        # Initial position of the snake's head
        self.head = Point(int(self._screen_width/2), int(self._screen_height/2))

        # Initial direction : right
        self.direction = direction.RIGHT

        # snake is defined as a list containing the coordinates of all the points of his body,
        # from head to tail
        self.snake = [self.head, 
                      Point(self.head.x - 1, self.head.y),
                      Point(self.head.x - 2, self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_new_food()
        self.done = False
        self.player_last_action = None
        self.last_distance_food = int(np.linalg.norm(np.array(self.food) - np.array(self.head)))

    def update_state(self, action):
        # stores the action
        self.player_last_action = action # [Left, Straight, Right], 1 if this is the chosen direction, 0 else

        ## 1. Move

        # update direction and head coordinates
        self._move(action) 
        # update the body of the snake, inserting the head as 1st element
        self.snake.insert(0, self.head) 

        ## 2. Check if game is over
        reward = self.get_reward_distance_food()
        self.done = False
        if self.is_collision():
            self.done = True
            reward = -100
            return reward, self.done, self.score 


        ## 3. Place new food if necessary
        if self.head == self.food:
            self.score += 1 # if snake eats food, score is +1
            reward = 10
            self._place_new_food()
            # do not pop the tail because the snakes grows +1 when it eats
        else:
            self.snake.pop() # pop the tail (snake has moved, last coordinate is not in the body anymore)

        return reward, self.done, self.score 
    
    def get_reward_distance_food(self):
        distance = int(np.linalg.norm(np.array(self.food) - np.array(self.head)))
        if distance < self.last_distance_food:
            reward = 1
        else :
            #old
            #reward = -5
            reward = -2
        self.last_distance_food = distance
        return reward


    def _move(self, action):
        """ 
        Inputs:
            action: int, 0 = Left, 1 = Straight, 2 = Right
        """
        # update direction
        if action == 0: # turns Left
            # anti clockwise change
            if self.direction == direction.LEFT:
                new_dir = direction.DOWN
            if self.direction == direction.RIGHT:
                new_dir = direction.UP
            if self.direction == direction.UP:
                new_dir = direction.LEFT
            if self.direction == direction.DOWN:
                new_dir = direction.RIGHT
        elif action == 1: # stays straight
            new_dir = self.direction # no change in direction
        else: # turns right
            # clockwise change
            if self.direction == direction.LEFT:
                new_dir = direction.UP
            if self.direction == direction.RIGHT:
                new_dir = direction.DOWN
            if self.direction == direction.UP:
                new_dir = direction.RIGHT
            if self.direction == direction.DOWN:
                new_dir = direction.LEFT

        self.direction = new_dir

        # update head coordinates
        x = self.head.x
        y = self.head.y
        if self.direction == direction.RIGHT:
            x += 1
        elif self.direction == direction.LEFT:
            x -= 1
        elif self.direction == direction.DOWN:
            y += 1 # y points down
        elif self.direction == direction.UP:
            y -= 1 

        self.head = Point(x,y)

    def _place_new_food(self):
        x = random.randint(2, self._screen_width-2)
        y = random.randint(2, self._screen_height-2)
        self.food = Point(x,y)
        # if the food appears in the body of snake, draw again
        if self.food in self.snake: 
            self._place_new_food()

    def is_collision(self, point=None):
        if point is None:
            point = self.head

        # check if it hits boundary
        if point.x >= self._screen_width or point.x <= 0 or point.y >= self._screen_height or point.y <= 0:
            return True
        
        # check if it hits its own body
        if point in self.snake[1:]:
            return True
        
        return False
        
