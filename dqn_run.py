import torch 
import matplotlib.pyplot as plt 
from text_snake_simple import TextSnakeEnvSimple
from dqn_functions.dqn_trainer import QTrainer

if __name__ == "__main__":
    env = TextSnakeEnvSimple(screen_size = (15, 10))    
    num_runs = 100        
    x_list = range(num_runs+1)
    print("Training...")
    qtrainer = QTrainer()
    dqn_score_list = [0]
    for j in range(num_runs):
        qtrainer.train(env, 500)
        obs, info = env.reset()
        r = None
        prev_tail = None
        done = False
        n_steps = 0
        while not done and n_steps < 5000:
            prediction = qtrainer.model(torch.tensor(obs, dtype=torch.float))
            action = qtrainer.argmax(prediction).item()
            obs, reward, done, _, info = env.step(action)
            n_steps += 1
            if j+1 == num_runs:
                env.render() 
            if done:
                break
        dqn_score_list.append(info["score"])
        env.close()
    plt.plot(x_list, dqn_score_list)
    plt.show()