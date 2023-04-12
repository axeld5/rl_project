import tqdm
from exp_sarsa_functions.exp_sarsa_utils import create_state_dict, run_episode
from exp_sarsa_functions.exp_sarsa_ag import ExpectedSarsaAgent
from text_snake_simple import TextSnakeEnvSimple

if __name__ == "__main__":
    env = TextSnakeEnvSimple(screen_size = (15, 10))
    state_dict = create_state_dict(n_bools=11)
    agent_info = {"num_actions": 3, "num_states": len(state_dict), "EPS_START": 0.9, "EPS_END": 0.05, 
            "EPS_DECAY": 5000, "step_size": 0.8, "discount": 0.99, "seed": 0}
    esarsa_agent = ExpectedSarsaAgent()
    esarsa_agent.agent_init(agent_info)
    num_runs = 1
    num_episodes = 5000
    for run in tqdm.tqdm(range(num_runs)):
        steps_done = 0
        for episode in tqdm.tqdm(range(num_episodes)):
            if episode+1 == num_episodes: 
                steps_done, converged = run_episode(env, esarsa_agent, episode, state_dict, steps_done, False, True)
                #put for debugging purposes
            elif (episode+1)%1000 == 0:
                steps_done, converged = run_episode(env, esarsa_agent, episode, state_dict, steps_done, True, False)
            else: 
                steps_done, converged = run_episode(env, esarsa_agent, episode, state_dict, steps_done, True, False)