
from PG import *

N = 6
WILD = True # In WILD variant, N-1 can count as any value

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
    NUM_ACTION = 2*N+1 # 0-(N-1) is single bet. N-(2N-1) is double bet. 2N is call
    TIME_HORIZON = 2*N

    CALL_ACTION = 2*N

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
        if self.actionHist[self.time-1] == self.CALL_ACTION:
            return self.actionHist[self.time-2]
        return self.actionHist[self.time-1]
    
    def getEndValue(self, types):
        # in the end state, currPlayer is the better, other player is the caller
        count = 0
        bet = self.lastBet()
        betVal = bet % N
        betQuan = bet // N + 1
        if WILD:
            if types[0].val in (betVal, N-1):
                count += 1
            if types[1].val in (betVal, N-1):
                count += 1
        else:
            if types[0].val == betVal:
                count += 1
            if types[1].val == betVal:
                count += 1
        posVal =  ((count >= betQuan)*2-1) * (self.currPlayer*2-1)
        return [-posVal, posVal]

    def getValidActions(self):
        actions = []
        for i in range(self.lastBet()+1, 2*N):
            actions.append(i)
        if self.time != 0:
            actions.append(self.CALL_ACTION)
        return actions
    
    def makeAction(self, action):
        assert action in self.getValidActions()
        self.actionHist[self.time] = action
        self.currPlayer = 1 - self.currPlayer
        self.time += 1
        if action == self.CALL_ACTION:
            self.endState = True
        elif (WILD and action == 2*N-1) or (not WILD and action >= N):
            self.endState = True
            self.currPlayer = 1 - self.currPlayer # action N+i means you end the game and bet double i
        elif self.time == self.TIME_HORIZON:
            self.endState = True