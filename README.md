# Reinforcement learning in Keras
![Tests](https://github.com/garethjns/reinforcement-learning-keras/workflows/Tests/badge.svg?branch=master) [![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=garethjns_reinforcement-learning-keras&metric=alert_status)](https://sonarcloud.io/dashboard?id=garethjns_reinforcement-learning-keras)

This repo aims to implement various reinforcement learning agents using Keras (tf==2.2.0) and sklearn, for use with OpenAI Gym environments.
  
# Planned agents
- [ ] Cart pole
    - [ ] Q-learning
        - [x] Linear Q learner (using sklearn.linear_model.SGDRegressor) 
        - [x] Deep Q leaner
        - [ ] Deep Q learner refinements
          - [x] Replay buffer
          - [ ] Unrolled Bellman
          - [x] Dueling architecture
          - [ ] Multiple environments
          - [ ] Double DQN
    - [ ] Policy gradient methods
        - [x] REINFORCE
        - [ ] Actor-critic
- [ ] Pong
   - [ ] Q-learning
        - [ ] Deep Q-learner 
   - [ ] Policy gradients
     - [ ] REINFORCE
     - [ ] Actor-critic

# General references
 - [Deep reinforcement learning hands-on, 2nd edition](https://www.amazon.co.uk/Deep-Reinforcement-Learning-Hands-optimization/dp/1838826998/), Maxim Lapan
 - [The Lazy Programmers'](https://lazyprogrammer.me/) courses: 
   - [Artificial Intelligence: Reinforcement learning in Python](https://www.udemy.com/course/artificial-intelligence-reinforcement-learning-in-python/learn/)
   - [Advanced AI: Deep reinforcement learning in Python](https://www.udemy.com/course/deep-reinforcement-learning-in-python/learn/)
   - [Cutting-edge AI: Deep reinforcement learning in Python](https://www.udemy.com/course/cutting-edge-artificial-intelligence/learn/)
 - Lilian Weng's overviews of reinforcement learning. I try and use the same terminology as used in these posts.
   - [A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)
   - [Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
 - Multiple Github repos and Medium posts on individual techniques - these are cited in context.
   
# Set-up
````bash
git clone 
cd reinforcement-learning-keras
pip install -r requirements.txt
```` 
  
# Cart-pole
Using cart-pole-v0 with step limit increased from 200 to 500.

## Linear Q learner
![Episode play example](https://github.com/garethjns/reinforcement-learning-keras/blob/master/images/LinearQAgent.gif) ![Convergence](https://github.com/garethjns/reinforcement-learning-keras/blob/master/images/LinearQAgent.png)  

Model:  
State -> model for action 1 -> value for action 1    
State -> model for action 2 -> value for action 2

This agent is based on [The Lazy Programmers](https://lazyprogrammer.me/) 2nd reinforcement learning course [implementation](https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl2/cartpole). It uses a separate SGDRegressor models for each action to estimate Q(a|s). Each step, the model for the selected action is updated using .partial_fit.  Action selection is off-policy and uses epsilon greedy; the selected either the argmax of action values, or a random action, depending on the current value of epsilon.

Environment observations are preprocessed in an sklearn pipeline that clips, scales, and creates features using RBFSampler.


### Run example
````bash
python3 -m agents.cart_pole.q_learning.linear_q_agent
````
or
````python
from agents.cart_pole.q_learning.components.epsilon_greedy import EpsilonGreedy
from agents.cart_pole.q_learning.linear_q_agent import LinearQAgent

agent = LinearQAgent(env_spec="CartPole-v0", 
                             eps=EpsilonGreedy(eps_initial=0.5, eps_min=0.01))
agent.train(verbose=True, render=True)
````

## Deep Q learner
![Episode play example](https://github.com/garethjns/reinforcement-learning-keras/blob/master/images/DQNAgent.gif) ![Convergence](https://github.com/garethjns/reinforcement-learning-keras/blob/master/images/DQNAgent.png)  

Model:  
State -> action model -> [value for action 1, value for action 2] 

A [deep Q learning](https://www.nature.com/articles/nature14236) agent that uses small neural network to approximate Q(s, a). It includes a replay buffer that allows for batched training updates, this is important for 2 reasons:
 - As this method is off-policy (the action is selected as argmax(action values)), it can train on data collected during previous episodes. This reduces correlation in the training data.
 - This is important for performance, especially when using a GPU. Calling multiple predict/train operations on single rows inside a loop is very inefficient. 

### Run example
````bash
python3 -m agents.cart_pole.q_learning.deep_q_agent 
````
or
````python
from agents.cart_pole.q_learning.components.epsilon_greedy import EpsilonGreedy
from agents.cart_pole.q_learning.components.replay_buffer import ReplayBuffer
from agents.cart_pole.q_learning.deep_q_agent import DeepQAgent

DeepQAgent.set_tf(256)  # Optional, limit tensorflow memory commitment to 256MB
agent = DeepQAgent(env_spec="CartPole-v0", 
                   eps=EpsilonGreedy(eps_initial=0.05, decay=0.002, eps_min=0.002),
                   replay_buffer=ReplayBuffer(buffer_size=200))
agent.train(verbose=True, render=True)
````

This agent uses two copies of its model:
 - One to predict the value of the next action, which us updated every episode step (with a batch sampled from the replay buffer)
 - One to predict value of the actions in the current and next state for calculating the discounted reward. This model is updated with the weights from the first model at the end of each episode.

## Dueling DQN
![Episode play example](https://github.com/garethjns/reinforcement-learning-keras/blob/master/images/DuelingDQNAgent.gif) ![Convergence](https://github.com/garethjns/reinforcement-learning-keras/blob/master/images/DuelingDQNAgent.png)  

State -> action model -> [value for action 1, value for action 2] 

The [dueling](https://arxiv.org/abs/1511.06581) version is exactly the same as the DQN, expect with slightly different model architecture. The second to last layer is split into two layers with the units=1 and units=n_actions. The idea is that the model might learn V(s) and action advantages (A(s)) separately, which can speed up convergence.  

The output of the network is still action values, however preceding layers are not fully connected; the values are now V(s) + A(s) and a subsequent Keras lambda layer is used to calculate the action advantages.
 
 ### Run example
````bash
python3 -m agents.cart_pole.q_learning.dueling_deep_q_agent
````
or
````python
from agents.cart_pole.q_learning.components.epsilon_greedy import EpsilonGreedy
from agents.cart_pole.q_learning.components.replay_buffer import ReplayBuffer
from agents.cart_pole.q_learning.dueling_deep_q_agent import DuelingDeepQAgent

DuelingDeepQAgent.set_tf(256)  # Optional, limit tensorflow memory commitment to 256MB
agent = DuelingDeepQAgent(env_spec="CartPole-v0", 
                          eps=EpsilonGreedy(eps_initial=0.05, decay=0.002, eps_min=0.002),
                          replay_buffer=ReplayBuffer(buffer_size=200))
agent.train(verbose=True, render=True)
````
 
## REINFORCE (policy gradient)
![Episode play example](https://github.com/garethjns/reinforcement-learning-keras/blob/master/images/REINFORCEAgent.gif) ![Convergence](https://github.com/garethjns/reinforcement-learning-keras/blob/master/images/REINFORCEAgent.png)  

Model:  
State -> model -> [probability of action 1, probability of action 2]  
Refs:  
https://github.com/Alexander-H-Liu/Policy-Gradient-and-Actor-Critic-Keras

[Policy gradient](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) models move the action selection policy into the model, rather than using argmax(action values). Model outputs are action probabilities rather than values (π(a|s), where π is the policy), making these methods inherently stochastic and removing the need for epsilon greedy action selection. 

This agent uses a small neural network to predict action probabilities given a state. Updates are done in a Monte-Carlo fashion - ie. using all steps from a single episode. This removes the need for a complex replay buffer (list.append() does the job). However as the method is on-policy it requires data from the current policy for training. This means training data can't be collected across episodes (assuming policy is updated at the end of each). This means the training data in each batch (episode) is highly correlated, which slows convergence.

This model doesn't use any scaling or clipping for environment pre-processing. For some reason, using the same pre-processing as with the DQN models prevents it from converging. The cart-pole environment can potentially return really huge values when sampling from the observation space, but these are rarely seen during training. It seems to be fine to pretend they don't exist, rather than scaling inputs based environment samples, as done with in the other methods.

````bash
python3 -m agents.cart_pole.policy_gradient.reinforce_agent
````
or
````python
from agents.cart_pole.policy_gradient.reinforce_agent import ReinforceAgent

ReinforceAgent.set_tf(256)  # Optional, limit tensorflow memory commitment to 256MB
agent = ReinforceAgent(env_spec="CartPole-v0")
agent.train(verbose=True, render=True)
````


# Mountain car

## Linear Q learner
![Episode play example](https://github.com/garethjns/reinforcement-learning-keras/blob/master/images/LinearQAgentMC.gif) ![Convergence](https://github.com/garethjns/reinforcement-learning-keras/blob/master/images/LinearQAgentMC.png)  

### Run example
````bash
python3 -m agents.mountain_car.q_learning.linear_q_agent
````
or
````python
from agents.mountain_car.q_learning.linear_q_agent import LinearQAgent

agent = LinearQAgent(env_spec="MountainCar-v0")
agent.train(verbose=True, render=True)
````