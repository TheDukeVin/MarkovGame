
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl

from nash import *

# from high import *
# from dicesum import *
from liarsdice import *

class Policy:
    def __init__(self, agentID):
        self.actionDist = {}
        for s in ALL_STATES():
            if s.endState or s.currPlayer != agentID: continue
            self.actionDist[s] = {}
            for t in ALL_TYPES():
                self.actionDist[s][t] = [0] * State.NUM_ACTION
                for a in s.getValidActions():
                    self.actionDist[s][t][a] = 1.0 / len(s.getValidActions())
    
    # def copy(self):
    #     p = Policy(self.agentID)
    #     for s in self.actionDist:
    #         for t in ALL_TYPES():
    #             for a in s.getValidActions():
    #                 p.actionDist[s][t][a] = self.actionDist[s][t][a]
    #     return p
    
    def copy(self, other):
        for s in other.actionDist:
            for t in ALL_TYPES():
                for a in s.getValidActions():
                    self.actionDist[s][t][a] = other.actionDist[s][t][a]
    
    def resetGradient(self):
        for s in self.actionDist:
            for t in ALL_TYPES():
                for a in s.getValidActions():
                    self.actionDist[s][t][a] = 0
    
    def __str__(self):
        output = ""
        for s in self.actionDist:
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

class Match:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def __hash__(self):
        return hash((self.a, self.b))
    
    def __eq__(self, other):
        return (self.a, self.b) == (other.a, other.b)

class PSRO:
    def __init__(self):
        pass
    
    def runPSRO(self, starting_queue, numIter, return_size):
        # self.policyQueue = []
        # for i in range(State.NUM_AGENT):
        #     self.policyQueue.append([starting_policy[i]])
        # self.matchVals = {Match(0, 0) : self.match(starting_policy)}
        self.policyQueue = starting_queue
        self.matchVals = {}
        self.returnPolicy = [[], []]
        start_time = time.time()
        self.fillMatch()
        for i in range(numIter):
            self.iterate(i >= numIter - return_size)
            print("Step " + str(i) + " TIME STAMP: " + str(time.time() - start_time))
    
    def iterate(self, submit=False):
        assert State.NUM_AGENT == 2
        nash = self.computeNash()
        # print(nash)

        responses = []
        vals = []
        for i in range(2):
            mix = self.getMixture(self.policyQueue[i], nash[i], i)
            response, val = self.getBestResponse(mix, 1-i)
            responses.append(response)
            vals.append(val)
            if submit:
                self.returnPolicy[i].append(mix)
        self.policyQueue[0].append(responses[1])
        self.policyQueue[1].append(responses[0])
        self.fillMatch()
        # added_nash = self.computeNash()
        # for i in range(2):
        #     mix = self.getMixture(self.policyQueue[i], added_nash[i], i)
        #     self.policyQueue[i][-1] = mix
        # self.fillMatch()
        print("Exploitability: " + str(vals))

    def fillMatch(self):
        # N = len(self.policyQueue[0])
        # for i in range(0, N):
        #     self.matchVals[Match(i, N-1)] = self.match([self.policyQueue[0][i], self.policyQueue[1][N-1]])
        #     if i != N-1:
        #         self.matchVals[Match(N-1, i)] = self.match([self.policyQueue[0][N-1], self.policyQueue[1][i]])
        for i in range(0, len(self.policyQueue[0])):
            for j in range(0, len(self.policyQueue[0])):
                if Match(i, j) not in self.matchVals:
                    self.matchVals[Match(i, j)] = self.match([self.policyQueue[0][i], self.policyQueue[1][j]])
    
    def getPayoff(self):
        N = len(self.policyQueue[0])
        payoff = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                payoff[i][j] = self.matchVals[Match(i, j)]
        return payoff

    def computeNash(self):
        p1, p2, _ = getNash(self.getPayoff())
        return [p1, p2]
    
    def getMixture(self, policies, dist, agentID):
        mix = Policy(agentID)
        for s in mix.actionDist:
            for t in ALL_TYPES():
                for a in s.getValidActions():
                    sum = 0
                    for i in range(len(policies)):
                        sum += policies[i].actionDist[s][t][a] * dist[i]
                    mix.actionDist[s][t][a] = sum
        return mix
    
    def getBestResponse(self, adv, agentID):
        response = Policy(agentID)
        sum = 0
        for t, p in initDist().items():
            val = self.bestResponseRec(agentID, State(), t, adv, response, oppDist(t))
            sum += p * val
        return response, sum
    
    def bestResponseRec(self, agentID, s : State, t : Type, otherPolicy : Policy, response : Policy, oppType):
        # Calculates best response, updates self.best
        # Returns value of best response

        # FOR 2-PLAYER ONLY
        if s.endState:
            sum = 0
            for ot, p in oppType.items():
                types = [None] * 2
                types[agentID] = t
                types[1-agentID] = ot
                sum += p * s.getEndValue(types)[agentID]
            return sum
        if s.currPlayer == agentID:
            maxVal = -1e+07
            bestAction = -1
            for a in s.getValidActions():
                new_s = s.copy()
                new_s.makeAction(a)
                candVal = self.bestResponseRec(agentID, new_s, t, otherPolicy, response, oppType)
                if maxVal < candVal:
                    maxVal = candVal
                    bestAction = a
                response.actionDist[s][t][a] = 0
            response.actionDist[s][t][bestAction] = 1
            return maxVal
        newOppType = {}
        for ot in oppType:
            newOppType[ot] = -1
        sum = 0
        for a in s.getValidActions():
            new_s = s.copy()
            new_s.makeAction(a)
            # Update distribution for opponent's type using Bayes' rule
            bayesScale = 0
            for ot in oppType:
                newOppType[ot] = oppType[ot] * otherPolicy.actionDist[s][ot][a]
                bayesScale += newOppType[ot]
            if bayesScale < 1e-07:
                continue
            assert bayesScale > 1e-07
            for ot in oppType:
                newOppType[ot] /= bayesScale
            sum += bayesScale * self.bestResponseRec(agentID, new_s, t, otherPolicy, response, newOppType)
        return sum
    
    def match(self, policies):
        sum = 0
        for t0, p0 in initDist().items():
            for t1, p1 in oppDist(t0).items():
                sum += p0 * p1 * self.matchRec(policies, State(), [t0, t1])
        return sum
    
    def matchRec(self, policies, s : State, types) -> float:
        if s.endState:
            return s.getEndValue(types)[1] # Agent 1 is maximizing.
        sum = 0
        for a in s.getValidActions():
            action_prob = policies[s.currPlayer].actionDist[s][types[s.currPlayer]][a]
            if action_prob < 1e-07:
                continue
            new_s = s.copy()
            new_s.makeAction(a)
            sum += action_prob * self.matchRec(policies, new_s, types)
        return sum

class IterPSRO:
    def __init__(self):
        self.psro = PSRO()
    
    def runIterPSRO(self, numIter, numPSRO):
        policy = [[Policy(0)], [Policy(1)]]
        for i in range(numIter):
            print("################# RUNNING PSRO ITERATION ####################")
            self.psro.runPSRO(policy, numPSRO, 5)
            policy = self.psro.returnPolicy

trainer = IterPSRO()
trainer.runIterPSRO(1, 300)

with open("psro.pickle", 'wb') as f:
    pickle.dump(trainer, f, pickle.HIGHEST_PROTOCOL)

# trainer = PSRO()
# trainer.runPSRO([Policy(0), Policy(1)], 30)

# size = len(trainer.getPayoff())
# data = np.array(trainer.getPayoff())

# cmap = mpl.colormaps['bwr']

# # fig, ax = plt.subplots(figsize=(6,6))

# plt.imshow(data, cmap=cmap, interpolation='none', extent=[0, size, size, 0])
# plt.colorbar()
# plt.xlabel("Active agent number")
# plt.ylabel("Adversary agent number")
# plt.title("Active reward")

# plt.savefig("eval.png")
