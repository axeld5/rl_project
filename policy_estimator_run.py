import torch.optim as optim

from text_snake_simple import TextSnakeEnvSimple 
from policy_estimator_functions.policy_estimator import policy_estimator, reinforce, make_greedy_run

if __name__ == "__main__":
    env = TextSnakeEnvSimple(screen_size = (15, 10))
    state, info = env.reset()
    num_runs = 1
    num_episodes = 100000
    for run in range(num_runs):
        neural_agent = policy_estimator(11, 3) 
        optimizer = optim.Adam(neural_agent.network.parameters(), lr=0.0001)
        total_rewards, converged = reinforce(env, neural_agent, optimizer, num_episodes=num_episodes)
        if converged:
            break
    make_greedy_run(env, neural_agent)
    env.close()