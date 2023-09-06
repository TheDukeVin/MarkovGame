
from PG import *

class State:
    NUM_AGENT = 2 # Agent 0 is minimizer, agent 1 maximizer
    NUM_ACTION = 2
    TIME_HORIZON = 10

    SAFE = 1
    SPIKE = 10
    spikeProb = 0.1

    def __init__(self):
        self.time = 0
        self.endState = False
        self.val = 0
    
    def __hash__(self):
        return hash((self.time, self.val))
    
    def __eq__(self, other):
        return self.time == other.time and self.val == other.val
    
    def __str__(self):
        return str(self.time) + ' ' + str(self.val[0])

    def copy(self):
        newState = State()
        newState.time = self.time
        newState.endState = self.endState
        newState.val = self.val
        return newState
    
    def getEndValue(self):
        if self.val > 0:
            return 1
        if self.val == 0:
            return 0
        if self.val < 0:
            return -1

    def makeAction(self, actions):
        # returns all future states as well as the associated probabilities
        states = []
        for c in [(0,0), (0,1), (1,0), (1,1)]:
            newState = self.copy()
            prob = 1
            for i in range(self.NUM_AGENT):
                prob *= self.spikeProb if c[i] == 1 else 1 - self.spikeProb
                if actions[i] == 0:
                    newState.val += self.SAFE * (2*i-1)
                else:
                    newState.val += c[i] * self.SPIKE * (2*i-1)
            newState.time += 1
            if newState.time == self.TIME_HORIZON:
                newState.endState = True
            states.append((newState, prob))
        return states