
from PG import *

N = 4

class Type:
    def __init__(self, val):
        self.val = val
    
    def __hash__(self):
        return hash(self.val)
    
    def __eq__(self, other):
        return self.val == other.val
    
    def __str__(self):
        return str(self.val)

def ALL_TYPES():
    types = []
    for i in range(0, N):
        types.append(Type(i))
    return types

def initDist():
    dist = {}
    for t in ALL_TYPES():
        dist[t] = 1.0 / N
    return dist

def oppDist(type):
    dist = {}
    for i in range(0, N):
        if i != type.val:
            dist[Type(i)] = 1.0 / (N-1)
    return dist

def sign(x):
    if x > 0: return 1
    if x == 0: return 0
    return -1

class State:
    NUM_AGENT = 2 # Agent 0 is minimizer, agent 1 maximizer
    NUM_ACTION = 2
    TIME_HORIZON = 2

    def __init__(self):
        self.time = 0
        self.endState = False
        self.currPlayer = 0

        self.actionHist = [-1] * self.TIME_HORIZON
    
    def __hash__(self):
        return hash(tuple(self.actionHist))
    
    def __eq__(self, other):
        return self.actionHist == other.actionHist
    
    def __str__(self):
        return str(self.actionHist)

    def copy(self):
        newState = State()
        newState.time = self.time
        newState.endState = self.endState
        newState.currPlayer = self.currPlayer
        newState.actionHist = list(self.actionHist)
        return newState
    
    def getEndValue(self, types):
        if self.actionHist[0] == 0:
            return sign(types[1].val - types[0].val)
        elif self.actionHist[1] == 0:
            return -1
        return 2 * sign(types[1].val - types[0].val)

    def getValidActions(self):
        return [0, 1]
    
    def makeAction(self, action):
        assert action in self.getValidActions()
        self.actionHist[self.time] = action
        self.currPlayer = 1 - self.currPlayer
        self.time += 1
        if action == 0:
            self.endState = True
        if self.time == self.TIME_HORIZON:
            self.endState = True