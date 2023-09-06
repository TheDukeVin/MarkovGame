
from high import *
# from dicesum import *
from nash import *

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

PSRO_MODE = "PSRO_MODE"
FICT_MODE = "FICT_MODE"

class PSRO:
    def __init__(self):
        self.policyBase = []
        for i in range(State.NUM_AGENT):
            self.policyBase.append([Policy()])
        self.mixture = Policy()
        self.best = Policy()
        self.N = 1
        self.returnPolicies = [None, None]
        self.mode = PSRO_MODE
        if self.mode == PSRO_MODE:
            self.valueMatch = [[0]]
    
    def getMixture(self, policies, mixture):
        for s in ALL_STATES():
            for t in ALL_TYPES():
                for a in s.getValidActions():
                    sum = 0
                    for i in range(len(mixture)):
                        sum += policies[i].actionDist[s][t][a] * mixture[i]
                    self.mixture.actionDist[s][t][a] = sum

    
    def bestMixtureResponse(self, agentID, policies, mixture):
        self.getMixture(policies, mixture)
        val = self.bestResponse(agentID, self.mixture)
        return val, self.best.copy()
    
    def iterate(self):
        if self.mode == PSRO_MODE:
            mix0, mix1, _ = getNash(np.array(self.valueMatch))
            # print(self.valueMatch)
        else:
            pass
        val0, best0 = self.bestMixtureResponse(0, self.policyBase[1], mix1)
        val1, best1 = self.bestMixtureResponse(1, self.policyBase[0], mix0)
        self.policyBase[0].append(best0)
        self.policyBase[1].append(best1)
        for i in range(self.N):
            self.valueMatch[i].append(0)
        self.valueMatch.append([0] * (self.N + 1))
        self.N += 1
        for i in range(self.N):
            for j in range(self.N):
                if i < self.N-1 and j < self.N-1:
                    continue
                match_pol = [self.policyBase[0][i], self.policyBase[1][j]]
                self.valueMatch[i][j] = self.match(match_pol)
        return val1 - val0
    
    def runPSRO(self):
        for i in range(10):
            exploitability = self.iterate()
            print("Exploitability: " + str(exploitability))
            if exploitability < 1e-07:
                break
        mix0, mix1, _ = getNash(np.array(self.valueMatch))
        # print(mix0, mix1)
        self.getMixture(self.policyBase[0], mix0)
        self.returnPolicies[0] = self.mixture.copy()
        self.getMixture(self.policyBase[1], mix1)
        self.returnPolicies[1] = self.mixture.copy()

    def match(self, policies):
        sum = 0
        for t0, p0 in initDist().items():
            for t1, p1 in oppDist(t0).items():
                # print("Matching " + str(t0) + ' ' + str(t1))
                # print(self.matchRec(policies, State(), [t0, t1]))
                sum += p0 * p1 * self.matchRec(policies, State(), [t0, t1])
        return sum
    
    def matchRec(self, policies, s : State, types) -> float:
        if s.endState:
            return s.getEndValue(types)
        sum = 0
        # print("Calculating on State " + str(s))
        for a in s.getValidActions():
            new_s = s.copy()
            new_s.makeAction(a)
            # print("Action Prob: " + str(policies[s.currPlayer].actionDist[s][types[s.currPlayer]][a]))
            sum += policies[s.currPlayer].actionDist[s][types[s.currPlayer]][a] * self.matchRec(policies, new_s, types)
        return sum
    
    def bestResponse(self, agentID, otherPolicy):
        sum = 0
        for t, p in initDist().items():
            # print(t)
            val = self.bestResponseRec(agentID, State(), t, otherPolicy, oppDist(t))
            # print(val)
            sum += p * val
        return sum
    
    def bestResponseRec(self, agentID, s : State, t : Type, otherPolicy : Policy, oppType):
        # Calculates best response, updates self.best
        # Returns value of best response

        # FOR 2-PLAYER ONLY
        # print("Calculating for " + str(s))
        if s.endState:
            sum = 0
            for ot, p in oppType.items():
                types = [None] * 2
                types[agentID] = t
                types[1-agentID] = ot
                sum += p * s.getEndValue(types)
            # print(sum)
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
            # print(maxVal)
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
        # print(sum)
        return sum
    
    def play(self):
        s = State()
        t0 = getType()
        t1 = getOppType(t0)
        print("Your type: " + str(t1))
        for i in range(State.TIME_HORIZON):
            print("State " + str(s))
            if s.currPlayer == 0:
                a = np.random.choice(State.NUM_ACTION, p=self.returnPolicies[0].actionDist[s][t0])
            else:
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


a = PSRO()
# print(a.match([Policy(), Policy()]))
# P1 = Policy()
# P1.actionDist[State()][Type(2)] = [0, 1]
# P1.actionDist[State()][Type(0)] = [1, 0]
# print(a.match([P1, Policy()]))

# P1 = Policy()
# a.bestResponseRec(1, State(), Type(0), P1, oppDist(1, Type(0)))
# print(a.bestResponse(1, P1))
# a.best.print()

# print(a.bestResponse(0, P1))
# a.best.print()
# a.bestResponseRec(0, State(), Type(2), P1, oppDist(0, Type(2)))

a.runPSRO()
# print("AGENT 0")
# a.returnPolicies[0].print()
# print("AGENT 1")
# a.returnPolicies[1].print()

f = open("policy.out", "w")
f.write(str(a.returnPolicies[0]))
f.write(str(a.returnPolicies[1]))
f.close()

# while True:
#     print("NEW GAME:")
#     a.play()