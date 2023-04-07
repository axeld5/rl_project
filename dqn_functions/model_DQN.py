import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import copy

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_sup_layers):
        super().__init__()
        self.first_layer = nn.Linear(input_size, hidden_size)
        self.mid_layers = [nn.ReLU()]
        for _ in range(n_sup_layers):
            self.mid_layers.append(nn.Linear(hidden_size, hidden_size))
            self.mid_layers.append(nn.ReLU())
        self.mid_layer = nn.Sequential(*self.mid_layers)
        self.last_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.mid_layer(self.first_layer(x))
        x = self.last_layer(x)
        return x


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, prediction_model):
        state = torch.tensor(state, dtype=torch.float)
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

