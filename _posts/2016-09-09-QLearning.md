---
layout: default 
title:  "Q-Learning"
date:   2016-09-08 10:07:35 -0700
categories: 
---

## Q-learning:
Q-learning is an online reinforcement learning process. Which means the agent learns from interacting with the world and observing the rewards from taken actions. This allows the agent to gain knowledge about which actions will create the most reward.


Starting interface for the agent:


```python
%matplotlib inline
import QLearner as ql
import mazeworld
import numpy as np
import random as rand
import time
import math
import matplotlib.pyplot as plt
```


```python
class Agent:
    def __init__(self, numberOfStates, numberOfActions):
        return
    
    def getAction(self, state):
        return 0 #action to be taken
    
    #Transition from state -> state_prime via the action with the reward. 
    #This is where the agent learns:
    def update(self, state, action, state_prime, reward):
        return
```

The agent above always takes action 0 and never learns anything. To improve this the agent must remember the rewards it has seen:


```python
class Agent:
    def __init__(self, numberOfStates, numberOfActions):
        #A Qvalue is just the value of an action give a state
        self.Qvalues = np.zeros((numberOfStates,numberOfActions))

    #This is used for visualization. It return the value of the state
    def getVal(self, state):
        return self.Qvalues[state][self.getAction(state)]
    
    #This will be used later
    def DoneLearning(self):
        return
    
    def getAction(self,state):
        return np.argmax(self.Qvalues[state])#Always take the action with the highest reward
    
    #Transition from state -> state_prime via the action with the reward. 
    #This is where the agent learns:
    def update(self,state, action, state_prime, reward):
        self.Qvalues[state][action] = reward
```

This is the next simplest agent. It stores the last seen reward for each state/action pair. To test this we need a enviroment. The enviroment will take the actions of the action and calculate the next state. 


```python
def update(state, action):
    return (state,0x0)
def goalState(state):
    return False
def executeEnviroment(agent, numActionsLeft):
        state = 0 #start state
        goal = False
        while(not goal and (numActionsLeft > 0)):
            numActionsLeft -=1
            action = agent.getAction(state)
            state_prime,reward = update(state, action)
            agent.update(state, action, state_prime, reward)
            state = state_prime
            goal = goalState(state)
            
def testAgent(agent, numRounds):
    restartCount = 100
    for i in range(0, numRounds):
        executeEnviroment(agent, restartCount)
testAgent(Agent(1,1), 10)

```

This enviroment does nothing but it shows the flow of simulation. Note: the enviroment terminates after a given number of steps. This will stop the execution if the agent does not find the goal state.  

To test the agent, there needs to be a functional enviroment. For this post it is mazeworld. The source code is not shown here but it is a simple maze with four actions (north, south, east, west). If the agent takes an action into a wall it stays in the same position. The rewards are always -1 unless you reach the goal state then it is 1. The start position is (1,1) and the goal is (size-1, size-1). 


```python
mazeworld.testAgent(agent = Agent(11**2, 4), size = 10, numberOfRounds=10)
```

    Map and the final route learned:
    


![png]({{ site.baseurl }}/assets/output_8_1.png)


    Final route legth:
    101
    State value heatmap:
    


![png]({{ site.baseurl }}/assets/output_8_3.png)


    Number of steps per trial:
    


![png]({{ site.baseurl }}/assets/{{ site.baseurl }}/assets/output_8_5.png)


    Execution Time:
    0.00499987602234
    

The graphs are showing that the agent did not learn how to navigate the maze. This is due to the agents update rule. It only takes into account the immediate reward of each action. This does not work for a maze because the goal is more then one action away. This is shown in the heatmap above as the start start (0,0) has a value of -1. So the agent think all actions are equally bad. 
### Furture rewards:
To fix this the agent need to account for furture rewards. For example, in this case reaching the goal states is worth 1 reward and all other actions are worth -1. So if the agent is two spots left of the goal. Then the optimal actions are to move right twice. With our current agent the learned state values (Q-values) are:  
(state1),(state2),(goal):  
(-1),    (1),     (N/A)  
This is due the the agent only saving the direct rewards. If state two is worth 1 then the value of state one should account for that. The simplest way to account for that is to add the value of the next state to the current states rewards:



```python
class Agent:
    def __init__(self, numberOfStates, numberOfActions):
        #A Qvalue is just the value of an action give a state
        self.Qvalues = np.zeros((numberOfStates,numberOfActions))

    #This is used for visualization. It return the value of the state
    def getVal(self, state):
        return self.Qvalues[state][self.getAction(state)]
    
    #This will be used later
    def DoneLearning(self):
        return
    
    def getAction(self,state):
        return np.argmax(self.Qvalues[state])#Always take the action with the highest reward
    
    #Transition from state -> state_prime via the action with the reward. 
    #This is where the agent learns:
    def update(self,state, action, state_prime, reward):
        self.Qvalues[state][action] = reward + self.getVal(state_prime)
```


```python
mazeworld.testAgent(agent = Agent(61**2, 4), size = 60, numberOfRounds=400)
```

    Map and the final route learned:
    


![png]({{ site.baseurl }}/assets/output_11_1.png)


    Final route legth:
    438
    State value heatmap:
    


![png]({{ site.baseurl }}/assets/output_11_3.png)


    Number of steps per trial:
    


![png]({{ site.baseurl }}/assets/output_11_5.png)


    Execution Time:
    2.6289999485
    

This agent learned how to navigate this maze. Also note that the number of step per trial decreases with the number of trials. This is due the the agent learning the maze. The reason this trivial agent worked is due to the maze dynamics being simple.

This is a special case of furture rewards. The general case is the state values is the reward plus the next states reward discounted by some constant from 0 to 1. This discount factor (Gamma) is used to make furture rewards not as valueable as present rewards. This is similar to value of furture money in finace.   
Depending on the enviroment the optimal gamma value varies. In this case Gamma = 1 is optimal. 



```python
class Agent:
    def __init__(self, numberOfStates, numberOfActions, gamma = 1):
        #A Qvalue is just the value of an action give a state
        self.Qvalues = np.zeros((numberOfStates,numberOfActions))
        self.gamma = 1

    #This is used for visualization. It return the value of the state
    def getVal(self, state):
        return self.Qvalues[state][self.getAction(state)]
    
    #This will be used later
    def DoneLearning(self):
        return
    
    def getAction(self,state):
        return np.argmax(self.Qvalues[state])#Always take the action with the highest reward
    
    #Transition from state -> state_prime via the action with the reward. 
    #This is where the agent learns:
    def update(self,state, action, state_prime, reward):
        self.Qvalues[state][action] = reward + self.gamma*self.getVal(state_prime)
```

### Execution policies
A execution policy is how the agent picks an action given a set of (action, value) pairs. Or the getAction function. 

The agent's current execution policy is a greedy policy. This means it always chooses the best possible action no matter what. This rarely works as the agent will not explore new actions. Which leads to sub optimal action choices. 

This agent current works because the initial Q-values are zero. This is higher then the highest possible reward. This will force the agent to explore any unvisited state before retrying states. This is called optimistic initialization. This works well in the case where the state space is small. It the state space is large the agent might never reach the goal by spending too much time exploring. 

There is a modification on the greedy policy call epsilon greedy. It is the same as the greedy policy but epsilon percent of the time it picks a random action. This enforces exploration. It is also one of the most widely used policies.



```python
class Agent:
    def __init__(self, numberOfStates, numberOfActions, gamma = 1, epsilon = 0.1):
        #A Qvalue is just the value of an action give a state
        self.Qvalues = np.zeros((numberOfStates,numberOfActions))
        self.gamma = 1
        self.epsilon = epsilon
        self.numberOfActions = numberOfActions

    #This is used for visualization. It return the value of the state
    def getVal(self, state):
        return self.Qvalues[state][self.greedyPolicy(state)]
    
    #After learning is done the agent should be purely greedy
    def DoneLearning(self):
        epsilon = 0
    
    def greedyPolicy(self, state):
        return np.argmax(self.Qvalues[state])
    
    def getAction(self,state):
        if(rand.random() < self.epsilon):
            return rand.randint(0, self.numberOfActions-1)
        else:
            return self.greedyPolicy(state)
    #Transition from state -> state_prime via the action with the reward. 
    #This is where the agent learns:
    def update(self,state, action, state_prime, reward):
        self.Qvalues[state][action] = reward + self.gamma*self.getVal(state_prime)
```


```python
mazeworld.testAgent(agent = Agent(61**2, 4), size = 60, numberOfRounds=400)
```

    Map and the final route learned:
    


![png]({{ site.baseurl }}/assets/output_16_1.png)


    Final route legth:
    402
    State value heatmap:
    


![png]({{ site.baseurl }}/assets/output_16_3.png)


    Number of steps per trial:
    


![png]({{ site.baseurl }}/assets/output_16_5.png)


    Execution Time:
    3.24099993706
    

This agent find the same results but converges slightly faster albeit more choatic. It also finds a slightly better route. The improvement is quite small due to the face that the maze world does not really need this type of exploration atleast at these sizes. 

### Update Rules
A update rule is how the agent updates its state or knowledge of the enviroment given new information. 

Currently the agent just replaces the q-value with the newly observed one. This works well if there is limited noise in the enviroment and it is deterministic. Both are the case in the maze world. 

The more general and robust rule is to mix the old value and the new value. This is done be a weighted average of the new value and the old value. The weight is called alpha and the old value is weighted with (1-alpha) and the new value is weighted with alpha.

The previous agent is the same as the one below if alpha is set to 1. 


```python
class Agent:
    def __init__(self, numberOfStates, numberOfActions, gamma = 1, epsilon = 0.1, alpha = 1):
        #A Qvalue is just the value of an action give a state
        self.Qvalues = np.zeros((numberOfStates,numberOfActions))
        self.gamma = 1
        self.alpha = 1
        self.epsilon = epsilon
        self.numberOfActions = numberOfActions

    #This is used for visualization. It return the value of the state
    def getVal(self, state):
        return self.Qvalues[state][self.greedyPolicy(state)]
    
    #After learning is done the agent should be purely greedy
    def DoneLearning(self):
        epsilon = 0
    
    def greedyPolicy(self, state):
        return np.argmax(self.Qvalues[state])
    
    def getAction(self,state):
        if(rand.random() < self.epsilon):
            return rand.randint(0, self.numberOfActions-1)
        else:
            return self.greedyPolicy(state)
    #Transition from state -> state_prime via the action with the reward. 
    #This is where the agent learns:
    def update(self,state, action, state_prime, reward):
        self.Qvalues[state][action] =self.getVal(state)*(1-self.alpha)+\
                                     self.alpha*(reward + self.gamma*self.getVal(state_prime))
```

To show the issue with alpha being one. The maze enviroment will now return a value of -10 for every action 10% of the time and -1 the other 90%. This will average to be a reward of -1 and should produce the same results.  


```python
mazeworld.testAgent(agent = Agent(61**2, 4), size = 60, numberOfRounds=500, randomReward=True)
```

    Map and the final route learned:
    


![png]({{ site.baseurl }}/assets/output_20_1.png)


    Final route legth:
    231
    State value heatmap:
    


![png]({{ site.baseurl }}/assets/output_20_3.png)


    Number of steps per trial:
    


![png]({{ site.baseurl }}/assets/output_20_5.png)


    Execution Time:
    4.73500013351
    

The agents execution policy did not converge and the execution time was increased due to maxing out the interation of more rounds. This is due to the agent loosing all its knowledge of a state when it gets the bad -10 value for a reward.

To make the execution more robust let's test the execution with alpha = .3


```python
mazeworld.testAgent(agent = Agent(61**2, 4, alpha = .3), size = 60, numberOfRounds=500, randomReward=True)
```

    Map and the final route learned:
    


![png]({{ site.baseurl }}/assets/output_22_1.png)


    Final route legth:
    1048
    State value heatmap:
    


![png]({{ site.baseurl }}/assets/output_22_3.png)


    Number of steps per trial:
    


![png]({{ site.baseurl }}/assets/output_22_5.png)


    Execution Time:
    4.84999990463
    

Notice the agent found the best path and covergence graph it smoother. 

For most enviroment there is noise and non deterministic behavior. Mean that an alpha less then 1 is needed.

## Q-learning
The final agent above is the full q-learning algorthim with a e-greedy policy. 


```python

```
