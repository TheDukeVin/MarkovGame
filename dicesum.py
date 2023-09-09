
from PG import *

N = 6

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
    return initDist()

class State:
    NUM_AGENT = 2 # Agent 0 is minimizer, agent 1 maximizer
    NUM_ACTION = 2*N # 0-(2N-2) is bet of that amount. 2N-1 is sell
    TIME_HORIZON = 2*N-1

    SELL_ACTION = 2*N-1

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
    
    def lastBet(self):
        if self.time == 0:
            return -1
        if self.actionHist[self.time-1] == self.SELL_ACTION:
            return self.actionHist[self.time-2]
        return self.actionHist[self.time-1]
    
    def getEndValue(self, types):
        return (types[1].val + types[0].val - self.lastBet()) * (self.currPlayer*2-1)

    def getValidActions(self):
        actions = []
        for i in range(self.lastBet()+1, 2*N-1):
            actions.append(i)
        if self.time != 0:
            actions.append(self.SELL_ACTION)
        return actions
    
    def makeAction(self, action):
        assert action in self.getValidActions()
        self.actionHist[self.time] = action
        self.currPlayer = 1 - self.currPlayer
        self.time += 1
        if action == self.SELL_ACTION:
            self.endState = True
        if action == 2*N-2:
            self.endState = True
            self.currPlayer = 1 - self.currPlayer # 2*N-2 means you end the game and buy at 2N-2
        if self.time == self.TIME_HORIZON:
            self.endState = True