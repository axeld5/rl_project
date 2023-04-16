import torch
import torch.nn as nn
import numpy as np 
import os 
import sys 
import time 

class policy_estimator():
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 256), 
            nn.LeakyReLU(), 
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_outputs),
            nn.Softmax(dim=-1))
    
    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] 
        for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

def reinforce(env, policy_estimator, optimizer, num_episodes=2000,
              batch_size=10, gamma=0.99):
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1
    action_space = np.arange(env.action_space.n)
    ep = 0
    converged = False
    while ep < num_episodes:
        s_0, info = env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        while done == False:
            action_probs = policy_estimator.predict(
                s_0).detach().numpy()
            action = np.random.choice(action_space, 
                    p=action_probs)
            s_1, r, done, _, info = env.step(action)
            
            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1
            if done:
                batch_rewards.extend(discount_rewards(
                    rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    batch_states = np.array(batch_states)
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(
                        batch_rewards)
                    action_tensor = torch.LongTensor(
                       batch_actions)
                    logprob = torch.log(
                        policy_estimator.predict(state_tensor))
                    selected_logprobs = reward_tensor * torch.gather(logprob, 1, action_tensor.unsqueeze(1)).squeeze()
                    loss = -selected_logprobs.mean()
                    loss.backward()
                    optimizer.step()
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1
                    
                avg_rewards = np.mean(total_rewards[-100:])
                print("\rEp: {} Average of last 100:" +   
                     "{:.2f}".format(
                     ep + 1, avg_rewards), end="")
                ep += 1
                last_score = info["score"]
        if last_score > 50:
            converged = True 
            break
                
    return total_rewards, converged

def make_greedy_run(env, agent, render=False):
    s_0, info = env.reset()
    done = False
    converged = False 
    num_steps = 0
    while done == False and num_steps < 5000:
        action_probs = agent.predict(s_0).detach().numpy()
        action = np.argmax(action_probs)
        s_1, reward, done, _, info = env.step(action)
        if render:
            env.render()
        s_0 = s_1
        num_steps += 1
        if info["score"] > 300:
          converged = True 
          break
    return converged, info["score"]