import torch.nn as nn
import matplotlib.pyplot as plt 
from text_snake_simple import TextSnakeEnvSimple
from stable_baselines3 import PPO




if __name__ == "__main__":    
    num_runs = 100
    x_list = range(num_runs+1)
    env = TextSnakeEnvSimple(screen_size = (15, 10))
    policy_kwargs = dict(activation_fn=nn.ReLU,
                net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, ent_coef=0.01)
    ppo_score_list = [0]
    for j in range(num_runs):
        model.learn(total_timesteps=500, progress_bar=True)
        vec_env = model.get_env() 
        obs = vec_env.reset()
        done = False 
        n_step = 0
        while not done and n_step < 5000: 
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            if j+1 == num_runs:
                env.render()
            n_step += 1
            if done: 
                break
        ppo_score_list.append(info[0]["score"])
    plt.plot(x_list, ppo_score_list)
    plt.show()
    