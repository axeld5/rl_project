import numpy as np
from tqdm import tqdm

from collections import deque
import random

from dqn_functions.model_DQN import *

class QTrainer:

    def __init__(self,
                 LR=0.001,
                 MAX_MEMORY=100000,
                 BATCH_SIZE=100,
                 GAMMA=1) -> None:
        self.model = Linear_QNet(11, 128, 3, 2)
        self.memory = deque(maxlen=MAX_MEMORY)
        self.gamma = GAMMA
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        self.BATCH_SIZE = BATCH_SIZE

    def train_short_memory(self, state, action, reward, next_state, done, prediction_model):
        self.train_step(state, action, reward, next_state, done, prediction_model)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self, prediction_model):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones, prediction_model)

    def choose_action(self, env, state, epsilon):
        random_int = random.uniform(0,1)
        if random_int > epsilon:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            action = self.argmax(prediction).item()
        else:
            action = env.action_space.sample()
        return action
    
    def argmax(self, prediction):
        """argmax with random tie-breaking
        """
        ties = np.where(prediction == torch.max(prediction))[0]
        return np.random.choice(ties)


    def train(self, env, n_episodes, epsilon_start = 1, clone_model_freq = 10, eps_decay=0.9, eps_min=0.05):
        score_list = []
        epsilon = epsilon_start

        for ep in tqdm(range(n_episodes)):

            if ep % clone_model_freq == 0:
                prediction_model = copy.deepcopy(self.model)

            state = env.reset()[0]
            done = False
            epsilon = max(epsilon*eps_decay, eps_min)
            while True:
            # get old state
                state_old = env._get_observation()

                # get move
                action = self.choose_action(env, state_old, epsilon)

                # perform move and get new state
                state_new, reward, done, _, info = env.step(action)
                
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

    def train_step(self, state, action, reward, next_state, done, prediction_model):
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float)
        next_state = np.array(next_state)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long) # in our implementation, one action is an int : 0, 1 or 2
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        

        # 1: predicted Q values with current state
        pred = self.model(state)

        # When generating target Q values, use a cloned copy of the learning network, 
        # and only update this clone every N steps (with N typically set at 1000 or 10000).
        # This prevents from looping over 1 single action

        # the target is the same as the prediction, apart from the index of the chosen action, that
        # is updated with our new knowledge
        target = pred.clone()
        for idx in range(len(done)): # several idx in the case of batch training
            # if game is done, the Q value is only the reward
            Q_new = reward[idx]
            if not done[idx]: # if not done, use Bellman equation to have the best update of Q (for
                                # this state and this action)
                Q_new = reward[idx] + self.gamma * torch.max(prediction_model(next_state[idx]))

            target[idx][action[idx].item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

    def argmax(self, prediction):
        """argmax with random tie-breaking
        """
        ties = np.where(prediction == torch.max(prediction))[0]
        return np.random.choice(ties)