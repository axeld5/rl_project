import torch.optim as optim
import matplotlib.pyplot as plt 

from text_snake_simple import TextSnakeEnvSimple 
from policy_estimator_functions.policy_estimator import policy_estimator, reinforce, make_greedy_run

if __name__ == "__main__":    
    num_runs = 100    
    x_list = range(num_runs+1)
    env = TextSnakeEnvSimple(screen_size = (15, 10))
    state, info = env.reset()
    reinf_score_list = [0]
    neural_agent = policy_estimator(11, 3)    
    optimizer = optim.Adam(neural_agent.network.parameters(), lr=0.0001)
    render = False 
    for j in range(num_runs): 
        num_episodes = 500
        total_rewards, converged = reinforce(env, neural_agent, optimizer, num_episodes=num_episodes)
        if j+1 == num_runs:
            render = True
        converged, score = make_greedy_run(env, neural_agent, render)
        reinf_score_list.append(score)
        env.close()
    plt.plot(x_list, reinf_score_list)
    plt.show()