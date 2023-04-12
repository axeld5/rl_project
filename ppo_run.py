import torch.nn as nn

from text_snake_simple import TextSnakeEnvSimple
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env

#


if __name__ == "__main__":
    env = TextSnakeEnvSimple(screen_size = (15, 10))
    policy_kwargs = dict(activation_fn=nn.ReLU,
                net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, ent_coef=0.01)
    model.learn(total_timesteps=100_000, progress_bar=True)
    vec_env = model.get_env() 
    obs = vec_env.reset()
    done = False 
    while not done: 
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        env.render()
        if done: 
            break