import os 
import sys 
import time

def run_episode(env, agent, episode, state_dict, steps_done, e_greedy=True, render=False):
  state, info = env.reset()
  action = agent.agent_start(state_dict[tuple(state)], steps_done) 
  observation, reward, done, _, info = env.step(action) 
  steps_done += 1
  t = 0
  converged = False
  r = None
  prev_tail = None      
  while not done:
    action = agent.agent_step(reward, state_dict[tuple(observation)], steps_done, e_greedy)
    steps_done += 1
    observation, reward, done, _, info = env.step(action) 
    t += 1         
    if render:
        env.render()
    if info["score"] > 300:
        converged = True   
        break
  if done:
    agent.agent_end(reward)
  return steps_done, converged, info["score"]

def create_state_dict(n_bools):
    state_dict = {}
    for i in range(2**n_bools):
        bin_rep = [int(x) for x in list('{0:0b}'.format(i))]
        state = [0]*(n_bools - len(bin_rep)) + bin_rep
        state_dict[tuple(state)] = i 
    return state_dict