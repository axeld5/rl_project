import numpy as np 
import torch.optim as optim

from text_snake_simple import TextSnakeEnvSimple 
from policy_estimator import policy_estimator, reinforce, make_greedy_run

if __name__ == "__main__":
    env = TextSnakeEnvSimple(screen_size = (15, 10))
    state, _, _, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n
    neural_agent = policy_estimator(11, 3) 
    optimizer = optim.Adam(neural_agent.network.parameters(), lr=0.001)
    num_episodes = 500000
    total_rewards = reinforce(env, neural_agent, optimizer, num_episodes=num_episodes)
    make_greedy_run(env, neural_agent)
    env.close()