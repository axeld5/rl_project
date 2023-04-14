# Agent description

**Actions** : int: 0, 1, 2 = Left, Straight, Right

**State** : Array of size 11 :

- Distance to Danger Left : int (0 if the point next to head is collision)
- Distance to Danger Straight : int
- Distance to Danger Right : int
- Last move direction is Left : Bool
- Last move direction is Right : Bool
- Last move direction is Up : Bool
- Last move direction is Down : Bool
- Food is left : Bool
- Food is right : Bool
- Food is up : Bool
- Food is down : Bool

**Rewards** :   
- Eat food : +50
- Crash wall or eat his tail = Collision : -100
- Else : If the snake is closer to the food than last step : +1 ; if the snake is not closer to the food than last step : -5

_NB: The training was not successful (snake gets trapped in infinite loops) when I tried the following rewards :_

- Eat food : +10
- Collision : -100
- Else 0

or : 

- Eat food : +10
- Collision : -100
- Else : max(20-distance to food, 0)

**Collision** : If the screen size is (10, 15), then there is a collision when the snake moves to a point where $x = 0$ or $x = 10$ or $y = 0$ or $y = 15$.

NB : Axis are (x,y), the 0 is at the top left corner of the screen ; y point down (=> going "UP" on the screen means y _decreases_), x points right --> it's easier for the print.

# DQN Model

We use a classic DQN model. Here are the steps within a **training episode** : 

- Get observation of the state
- Choose an action using an epsilon-greedy policy. With probability $\varepsilon$, a random action is chosen. With probability $1 - \varepsilon$, the Neural Network is applied to compute the action-values of this state and to choose the best action (argmax of predicted Q values).
- The new state and the reward are collected, based on the chosen action $a$.
- Based on the new state and reward, the target action-value corresponding to $a$ is computed (using TD approximation)
- The Neural Network's loss can be computed (difference between predicted action-value and target action-value), and the gradients are backpropagated.
- Redo until game over, and reset states for the next episode

If we only use these steps of training, the snake often gets stuck into an infinite loop. The problem is that the network is receiving its own biased outputs as targets, and this typically results in an agent that fixates on a single default action, because it has learned an inflated action value of it.

The usual solution is to ([source](https://ai.stackexchange.com/questions/13202/reinforcement-learning-to-play-snake-network-seems-to-not-get-trained-at-all)) : 
- Use an "experience replay table". ie we store all $s, a, r, s'$ (state, action, reward, next state) in memory. Once in a while (the frequency here is set to every 10 episodes), we take a random sample (here mini batch size is 100) of those stored items, and train the network once on this mini batch. This is a kind of "long term memory".
- Use a "target network". When generating target Q values, use a cloned copy of the learning network, and only update this clone every N steps (with N here set to 10 episodes).

# Expected SARSA Model

We use a classic SARSA model. Here are the steps within a **training episode** : 

- Get observation of the state
- Choose an action using an epsilon-greedy policy. With probability $\varepsilon$, a random action is chosen. With probability $1 - \varepsilon$, the SARSA algorithm is used to choose the next action (argmax of predicted SARSA Q values) . The value of epsilon is made to decay and reach progressively 0 to make room for exploration early and make the algorithm more deterministic later in stages.
- The SARSA Q values are updates based on the SARSA Algorithm 
- Redo until game over, and reset states for the next episode

# Policy estimator functions
