import os, sys
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import math 
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time


from collections import namedtuple, deque
from itertools import count
from abc import ABCMeta, abstractmethod
from tqdm import tqdm

class BaseAgent:
    """Implements the agent for an RL-Glue environment.
    Note:
        agent_init, agent_start, agent_step, agent_end, agent_cleanup, and
        agent_message are required methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def agent_init(self, agent_info= {}):
        """Setup for the agent called when the experiment first starts."""

    @abstractmethod
    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """

    @abstractmethod
    def agent_cleanup(self):
        """Cleanup done after the agent ends."""

    @abstractmethod
    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """

class ExpectedSarsaAgent(BaseAgent):
    def agent_init(self, agent_init_info):
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.EPS_START = agent_init_info["EPS_START"]
        self.EPS_END = agent_init_info["EPS_END"]
        self.EPS_DECAY = agent_init_info["EPS_DECAY"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])
        self.q = np.zeros((self.num_states, self.num_actions))

    def agent_start(self, state, steps_done, e_greedy=True):
        self.epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon and e_greedy:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_step(self, reward, state, steps_done, e_greedy=True):
        self.epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)
        current_q = self.q[state,:]
        if self.rand_generator.rand() < self.epsilon and e_greedy:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        policy_s = np.zeros(self.num_actions)
        policy_s[:] = self.epsilon/self.num_actions
        best_a = np.argmax(current_q)
        policy_s[best_a] = self.epsilon/self.num_actions + 1 - self.epsilon 
        exp_sarsa_term = sum([policy_s[a] * current_q[a] for a in range(self.num_actions)])
        self.q[self.prev_state, self.prev_action] += self.step_size*(reward + self.discount*exp_sarsa_term 
                                                                     - self.q[self.prev_state, self.prev_action])
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_end(self, reward):
        self.q[self.prev_state, self.prev_action] += self.step_size*(reward 
                                                                     - self.q[self.prev_state, self.prev_action])
        
    def argmax(self, q_values):
        top = float("-inf")
        ties = []
        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []
            if q_values[i] == top:
                ties.append(i)
        return self.rand_generator.choice(ties)