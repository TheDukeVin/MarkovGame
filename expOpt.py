
import pickle
import time

from nash import *

# from high import *
# from dicesum import *
from liarsdice import *

fileName = "liarsdice_wild.pickle"

"""
rsync -r MarkovGame kevindu@login.rc.fas.harvard.edu:./MultiagentSnake
rsync -r kevindu@login.rc.fas.harvard.edu:./MultiagentSnake/LSTM/net.out MarkovGame
"""

class Policy:
    def __init__(self):
        self.actionDist = {}
        for s in ALL_STATES():
            self.actionDist[s] = {}
            for t in ALL_TYPES():
                self.actionDist[s][t] = [0] * State.NUM_ACTION
                for a in s.getValidActions():
                    self.actionDist[s][t][a] = 1.0 / len(s.getValidActions())
    
    def copy(self):
        p = Policy()
        for s in ALL_STATES():
            for t in ALL_TYPES():
                for a in s.getValidActions():
                    p.actionDist[s][t][a] = self.actionDist[s][t][a]
        return p
    
    def clone(self, other):
        for s in ALL_STATES():
            for t in ALL_TYPES():
                for a in s.getValidActions():
                    self.actionDist[s][t][a] = other.actionDist[s][t][a]
    
    def __str__(self):
        output = ""
        for s in ALL_STATES():
            output += "STATE: " + str(s) + "\n"
            for t in ALL_TYPES():
                output += "Type: " + str(t) + "\nAction: " + str(self.actionDist[s][t]) + "\n"
        return output

def ALL_STATES(s=State()):
    states = {s}
    if s.endState:
        return states
    for a in s.getValidActions():
        new_s = s.copy()
        new_s.makeAction(a)
        states.update(ALL_STATES(new_s))
    return states

def getType():
    typeDist = initDist()
    types = list(typeDist.keys())
    prob = []
    for i in range(0, len(types)):
        prob.append(typeDist[types[i]])
    index = np.random.choice(len(types), p=prob)
    return types[index]

def getOppType(type):
    typeDist = oppDist(type)
    types = list(typeDist.keys())
    prob = []
    for i in range(0, len(types)):
        prob.append(typeDist[types[i]])
    index = np.random.choice(len(types), p=prob)
    return types[index]

evalPeriod = 10
MERGE = True

class ExpOpt:
    def __init__(self):
        self.best = Policy()
        self.adjust = Policy()
        self.policies = [Policy(), Policy()]
        self.lastExploit = 1e+07
        self.learnRate = 0.1
        self.annealRate = 0.5
        self.policyStore = []
        if MERGE:
            for i in range(evalPeriod):
                self.policyStore.append([Policy(), Policy()])
    
    def correct(self, policy, validActions):
        done = False
        while not done:
            sum = 0
            for a in validActions:
                sum += policy[a]
            for a in validActions:
                policy[a] += (1-sum) / len(validActions)
            done = True
            for a in validActions:
                if policy[a] < 0:
                    done = False
                    policy[a] = 0
                    validActions.remove(a)
    
    def incPolicy(self, agentID, adjust, exploit):
        for s in ALL_STATES():
            for t in ALL_TYPES():
                for a in s.getValidActions():
                    self.policies[agentID].actionDist[s][t][a] += adjust.actionDist[s][t][a] * self.learnRate # (self.learnRate * exploit)
                self.correct(self.policies[agentID].actionDist[s][t], s.getValidActions())
    
    def iterate(self):
        val0 = self.bestResponse(1, self.policies[0])
        self.getAdjust(0)
        A0 = self.adjust.copy()
        val1 = self.bestResponse(0, self.policies[1])
        self.getAdjust(1)
        A1 = self.adjust.copy()
        print((val0, val1))

        exploit = val0 - val1
        self.incPolicy(0, A0, exploit)
        self.incPolicy(1, A1, exploit)

        return exploit
    
    def merge(self):
        print("MERGING...")
        for agentID in (0, 1):
            for s in ALL_STATES():
                for t in ALL_TYPES():
                    for a in s.getValidActions():
                        self.policies[agentID].actionDist[s][t][a] = 0
                        for i in range(evalPeriod):
                            self.policies[agentID].actionDist[s][t][a] += self.policyStore[i][agentID].actionDist[s][t][a]
                        self.policies[agentID].actionDist[s][t][a] /= evalPeriod
    
    def runExpOut(self, numIter):
        start_time = time.time()
        for i in range(numIter):
            exploitability = self.iterate()
            if MERGE:
                self.policyStore[i%10][0].clone(self.policies[0])
                self.policyStore[i%10][1].clone(self.policies[1])
                if i%evalPeriod == evalPeriod-1:
                    self.merge()
            if i%evalPeriod == 0:
                with open(fileName, 'wb') as f:
                    pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
                if exploitability > self.lastExploit:
                    self.learnRate *= self.annealRate
                self.lastExploit = exploitability
                print("Step: " + str(i) + " Exploitability: " + str(exploitability) + " Learn Rate: " + str(self.learnRate) + " Time Stamp: " + str(time.time() - start_time))
            if exploitability < 1e-07:
                break
    
    def bestResponse(self, agentID, otherPolicy):
        sum = 0
        for t, p in initDist().items():
            val = self.bestResponseRec(agentID, State(), t, otherPolicy, oppDist(t))
            sum += p * val
        return sum
    
    def bestResponseRec(self, agentID, s : State, t : Type, otherPolicy : Policy, oppType):
        # Calculates best response, updates self.best
        # Returns value of best response

        # FOR 2-PLAYER ONLY
        if s.endState:
            sum = 0
            for ot, p in oppType.items():
                types = [None] * 2
                types[agentID] = t
                types[1-agentID] = ot
                sum += p * s.getEndValue(types)
            return sum
        if s.currPlayer == agentID:
            # AGENT 0 MINIMIZES, AGENT 1 MAXIMIZES
            # here, scale maxVal as value for each player
            maxVal = -1e+07
            bestAction = -1
            for a in s.getValidActions():
                new_s = s.copy()
                new_s.makeAction(a)
                candVal = self.bestResponseRec(agentID, new_s, t, otherPolicy, oppType) * (agentID*2-1)
                if maxVal < candVal:
                    maxVal = candVal
                    bestAction = a
                self.best.actionDist[s][t][a] = 0
            self.best.actionDist[s][t][bestAction] = 1
            maxVal *= (agentID*2-1)
            return maxVal
        newOppType = {}
        for ot in oppType:
            newOppType[ot] = -1
        sum = 0
        for a in s.getValidActions():
            action_prob = 0
            for ot, p in oppType.items():
                action_prob += otherPolicy.actionDist[s][ot][a] * p
            if action_prob < 1e-07:
                continue
            new_s = s.copy()
            new_s.makeAction(a)
            # Update distribution for opponent's type using Bayes' rule
            bayesScale = 0
            for ot in oppType:
                newOppType[ot] = oppType[ot] * otherPolicy.actionDist[s][ot][a]
                bayesScale += newOppType[ot]
            assert bayesScale > 1e-07
            for ot in oppType:
                newOppType[ot] /= bayesScale
            sum += action_prob * self.bestResponseRec(agentID, new_s, t, otherPolicy, newOppType)
        return sum
    
    def getAdjust(self, agentID):
        for t, _ in initDist().items():
            self.getAdjustRec(agentID, State(), t, oppDist(t))

    def getAdjustRec(self, agentID, s : State, t : Type, oppType):
        if s.endState:
            sum = 0
            for ot, p in oppType.items():
                types = [None] * 2
                types[agentID] = t
                types[1-agentID] = ot
                sum += p * s.getEndValue(types)
            return sum
        if s.currPlayer == agentID:
            sum = 0
            actionVals = {}
            for a in s.getValidActions():
                new_s = s.copy()
                new_s.makeAction(a)
                actionVals[a] = self.getAdjustRec(agentID, new_s, t, oppType)
                sum += self.policies[agentID].actionDist[s][t][a] * actionVals[a]
            for a in s.getValidActions():
                # if (actionVals[a] - sum) * (2*agentID-1) > 0:
                #     self.adjust.actionDist[s][t][a] = 1
                # else:
                #     self.adjust.actionDist[s][t][a] = -1
                self.adjust.actionDist[s][t][a] = (actionVals[a] - sum) * (2*agentID-1)
        newOppType = {}
        for ot in oppType:
            newOppType[ot] = -1
        sum = 0
        for a in s.getValidActions():
            action_prob = 0
            for ot, p in oppType.items():
                action_prob += self.best.actionDist[s][ot][a] * p
            if action_prob < 1e-07:
                continue
            new_s = s.copy()
            new_s.makeAction(a)
            # Update distribution for opponent's type using Bayes' rule
            bayesScale = 0
            for ot in oppType:
                newOppType[ot] = oppType[ot] * self.best.actionDist[s][ot][a]
                bayesScale += newOppType[ot]
            assert bayesScale > 1e-07
            for ot in oppType:
                newOppType[ot] /= bayesScale
            sum += action_prob * self.getAdjustRec(agentID, new_s, t, newOppType)
        return sum
    
    def play(self, reveal=False):
        s = State()
        t0 = getType()
        t1 = getOppType(t0)
        print("Your type: " + str(t1))
        if reveal:
            print("Opponent type: " + str(t0))
        for i in range(State.TIME_HORIZON):
            print("State " + str(s))
            if s.currPlayer == 0:
                if reveal:
                    print("Opponent policy: " + str(self.returnPolicies[0].actionDist[s][t0]))
                a = np.random.choice(State.NUM_ACTION, p=self.returnPolicies[0].actionDist[s][t0])
            else:
                if reveal:
                    print("Your policy: " + str(self.returnPolicies[1].actionDist[s][t1]))
                while True:
                    try:
                        a = int(input("Enter Action: "))
                        if a in s.getValidActions():
                            break
                    except ValueError:
                        pass
                    print("Please enter valid action, i.e. 0 to " + str(State.NUM_ACTION-1))
            s.makeAction(a)
            print("Action " + str(a))
            if s.endState:
                break
        print("State " + str(s))
        print("Opponent type " + str(t0))
        print("Value " + str(s.getEndValue([t0, t1])))
        return s.getEndValue([t0, t1])
