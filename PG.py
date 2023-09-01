
import numpy as np
import random

learnRate = 0.01
meanRate = 0.001

class PG:
    def __init__(self, numActions):
        self.N = numActions
        self.weights = np.zeros(numActions)
        self.cumProb = np.ones(numActions) / numActions
        self.T = 0
        self.avgReward = 0
    def sample(self):
        self.prob = np.exp(self.weights)
        self.prob /= np.sum(self.prob)
        self.action = np.random.choice(self.N, p=self.prob)
        return self.action
    def update(self, reward):
        action = np.zeros(self.N)
        action[self.action] = 1
        self.weights += (action - self.prob) * learnRate * (reward - self.avgReward)
        self.cumProb = (self.cumProb * self.T + self.prob) / (self.T + 1)
        self.T += 1
        self.avgReward += (reward - self.avgReward) * meanRate

def banditTest():
    N = 5
    a = PG(N)
    probs = np.random.uniform(size=N)
    print("True rewards: ")
    print(probs)
    for i in range(10000):
        act = a.sample()
        a.update(np.random.uniform() < probs[act])
    print("Learned:")
    print(a.prob)

# banditTest()

def matrixTest():
    N = 3
    payoff = [
        [1,-1,-1],
        [0,1,-1],
        [0,0,1]
    ]
    a = PG(N)
    b = PG(N)
    for i in range(100000):
        act1 = a.sample()
        act2 = b.sample()
        a.update(payoff[act1][act2])
        b.update(-payoff[act1][act2])
    print(a.cumProb)
    print(b.cumProb)

# matrixTest()