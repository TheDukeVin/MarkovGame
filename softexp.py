
import pickle
import time

from nash import *

# from high import *
# from dicesum import *
from liarsdice import *

TYPES = ALL_TYPES()
NUM_TYPES = len(TYPES)

# borderSpace = 1e-05

class Data:
    def __init__(self, N):
        self.N = N
        self.val = [0] * N
        self.grad = [0] * N
    def resetGrad(self):
        for i in range(self.N):
            self.grad[i] = 0
    def __str__(self):
        return "Val: " + str(self.val) + "\nGrad: " + str(self.grad)

class Policy:
    def __init__(self, agentID):
        self.actionDist = {}
        for s in ALL_STATES():
            if s.endState or s.currPlayer != agentID: continue
            self.actionDist[s] = {}
            for t in ALL_TYPES():
                self.actionDist[s][t] = Data(State.NUM_ACTION)
                for a in s.getValidActions():
                    self.actionDist[s][t].val[a] = 1.0 / len(s.getValidActions())
    
    def copy(self, other):
        for s in other.actionDist:
            for t in ALL_TYPES():
                for a in s.getValidActions():
                    self.actionDist[s][t].val[a] = other.actionDist[s][t].val[a]
                    self.actionDist[s][t].grad[a] = other.actionDist[s][t].grad[a]
    
    def resetGradient(self):
        for s in self.actionDist:
            for t in ALL_TYPES():
                self.actionDist[s][t].resetGrad()
    
    def printGrad(self):
        for s in self.actionDist:
            for t in ALL_TYPES():
                print("State: " + str(s) + " Type: " + str(t))
                print(self.actionDist[s][t].grad)
    
    def __str__(self):
        output = ""
        for s in self.actionDist:
            output += "STATE: " + str(s) + "\n"
            for t in ALL_TYPES():
                output += "Type: " + str(t) + "\nAction: " + str(self.actionDist[s][t].val) + "\n"
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

class SoftExp:
    # Calculates soft exploitability of policy as well as
    # its gradient with respect to policy
    def __init__(self, agentID):
        self.softMult = 1
        self.borderPenalty = 1
        self.agentID = agentID
        self.policy = Policy(agentID) # Given active agent
        self.cumulantGradient = Policy(agentID) # cumulant gradient for active agent
        self.posteriorType = {} # Posterior type of active agent
        self.value = {} # Adversary value
        self.action_dist = {} # Posterior action distribution of active agent
        self.adv_action_dist = {} # Softmax action dist of adversary agent
        for s in ALL_STATES():
            self.posteriorType[s] = Data(NUM_TYPES)
            self.value[s] = Data(1)
            self.action_dist[s] = Data(State.NUM_ACTION)
            self.adv_action_dist[s] = Data(State.NUM_ACTION)
    
    def calcPosterior(self, s : State, a, new_s : State, advType : Type):
        if s.currPlayer == self.agentID:
            likelihood = np.zeros(NUM_TYPES)
            for i in range(0, NUM_TYPES):
                likelihood[i] = self.policy.actionDist[s][TYPES[i]].val[a] * self.posteriorType[s].val[i]
            self.action_dist[s].val[a] = np.sum(likelihood)
            for i in range(0, NUM_TYPES):
                self.posteriorType[new_s].val[i] = likelihood[i] / self.action_dist[s].val[a]
        else:
            for i in range(0, NUM_TYPES):
                self.posteriorType[new_s].val[i] = self.posteriorType[s].val[i]
    
    def calcPosteriorGrad(self, s : State, a, new_s : State, advType : Type):
        if s.currPlayer == self.agentID:
            for i in range(0, NUM_TYPES):
                self.policy.actionDist[s][TYPES[i]].grad[a] += (
                    self.posteriorType[new_s].grad[i] * 
                    self.posteriorType[s].val[i] /
                    self.action_dist[s].val[a]
                )
                self.posteriorType[s].grad[i] += (
                    self.posteriorType[new_s].grad[i] * 
                    self.policy.actionDist[s][TYPES[i]].val[a] /
                    self.action_dist[s].val[a]
                )
                self.action_dist[s].grad[a] -= (
                    self.posteriorType[new_s].grad[i] * 
                    self.posteriorType[new_s].val[i] /
                    self.action_dist[s].val[a]
                )
                # scale = self.posteriorType[new_s].val[i] * self.posteriorType[new_s].grad[i]
                # self.policy.actionDist[s][TYPES[i]].grad[a] += scale / self.policy.actionDist[s][TYPES[i]].val[a]
                # self.posteriorType[s].grad[i] += scale / self.posteriorType[s].val[i]
                # self.action_dist[s].grad[a] -= scale / self.action_dist[s].val[a]
            for i in range(0, NUM_TYPES):
                self.policy.actionDist[s][TYPES[i]].grad[a] += self.action_dist[s].grad[a] * self.posteriorType[s].val[i]
                self.posteriorType[s].grad[i] += self.action_dist[s].grad[a] * self.policy.actionDist[s][TYPES[i]].val[a]
        else:
            for i in range(0, NUM_TYPES):
                self.posteriorType[s].grad[i] += self.posteriorType[new_s].grad[i] # * self.adv_action_dist[s].val[a]

    def calcEndValue(self, s : State, advType : Type):
        sum = 0
        for i in range(0, NUM_TYPES):
            types = [None] * 2
            types[self.agentID] = TYPES[i]
            types[1-self.agentID] = advType
            sum += self.posteriorType[s].val[i] * s.getEndValue(types)[1-self.agentID]
        self.value[s].val[0] = sum
    
    def calcEndGrad(self, s : State, advType : Type):
        for i in range(0, NUM_TYPES):
            types = [None] * 2
            types[self.agentID] = TYPES[i]
            types[1-self.agentID] = advType
            self.posteriorType[s].grad[i] += s.getEndValue(types)[1-self.agentID] * self.value[s].grad[0]

    def calcValue(self, s : State, advType : Type):
        if s.currPlayer == self.agentID:
            sum = 0
            for a in s.getValidActions():
                new_s = s.copy()
                new_s.makeAction(a)
                sum += self.action_dist[s].val[a] * self.value[new_s].val[0]
            self.value[s].val[0] = sum
        else:
            values = np.zeros(State.NUM_ACTION)
            for a in s.getValidActions():
                new_s = s.copy()
                new_s.makeAction(a)
                values[a] = self.value[new_s].val[0]
            exp_values = np.exp((values - np.max(values)) * self.softMult)
            sum = 0
            for a in s.getValidActions():
                self.adv_action_dist[s].val[a] = exp_values[a] / np.sum(exp_values)
                sum += self.adv_action_dist[s].val[a] * values[a]
            self.value[s].val[0] = sum

    def calcValueGrad(self, s : State, advType : Type):
        if s.currPlayer == self.agentID:
            for a in s.getValidActions():
                new_s = s.copy()
                new_s.makeAction(a)
                self.action_dist[s].grad[a] += self.value[new_s].val[0] * self.value[s].grad[0]
                self.value[new_s].grad[0] += self.action_dist[s].val[a] * self.value[s].grad[0]
        else:
            for a in s.getValidActions():
                new_s = s.copy()
                new_s.makeAction(a)
                self.value[new_s].grad[0] += (
                    self.adv_action_dist[s].val[a] * 
                    (1 + self.softMult * (self.value[new_s].val[0] - self.value[s].val[0])) *
                    self.value[s].grad[0]
                )


    def forwardPost(self, s : State, advType : Type):
        if s.endState:
            return
        for a in s.getValidActions():
            new_s = s.copy()
            new_s.makeAction(a)
            self.calcPosterior(s, a, new_s, advType)
            self.forwardPost(new_s, advType)
    
    def backwardPost(self, s : State, advType : Type):
        if s.endState:
            return
        for a in s.getValidActions():
            new_s = s.copy()
            new_s.makeAction(a)
            self.backwardPost(new_s, advType)
            self.calcPosteriorGrad(s, a, new_s, advType)
    
    def forwardValue(self, s : State, advType : Type):
        if s.endState:
            self.calcEndValue(s, advType)
            # print(str(s) + " value: " + str(self.value[s].val[0]))
            return
        for a in s.getValidActions():
            new_s = s.copy()
            new_s.makeAction(a)
            self.forwardValue(new_s, advType)
        self.calcValue(s, advType)
        # print(str(s) + " value: " + str(self.value[s].val[0]))
    
    def backwardValue(self, s : State, advType : Type):
        if s.endState:
            self.calcEndGrad(s, advType)
            return
        self.calcValueGrad(s, advType)
        for a in s.getValidActions():
            new_s = s.copy()
            new_s.makeAction(a)
            self.backwardValue(new_s, advType)
        # print("State: " + str(s) + " Value Gradient: " + str(self.value[s].grad[0]))
    
    def forwardPass(self, advType : Type):
        typeDist = oppDist(advType)
        for i in range(NUM_TYPES):
            self.posteriorType[State()].val[i] = typeDist[TYPES[i]]
        self.forwardPost(State(), advType)
        self.forwardValue(State(), advType)
    
    def backwardPass(self, advType: Type, prob : float):
        self.value[State()].grad[0] = prob
        self.backwardValue(State(), advType)
        self.backwardPost(State(), advType)
    
    def resetGradient(self):
        self.policy.resetGradient()
        for s in ALL_STATES():
            self.posteriorType[s].resetGrad()
            self.value[s].resetGrad()
            self.action_dist[s].resetGrad()
            self.adv_action_dist[s].resetGrad()
    
    def accGradient(self):
        for s in self.policy.actionDist:
            for t in ALL_TYPES():
                for a in s.getValidActions():
                    self.cumulantGradient.actionDist[s][t].grad[a] += self.policy.actionDist[s][t].grad[a]
    
    def test(self):
        for i in range(NUM_TYPES):
            advType = TYPES[i]
            self.forwardPass(advType)
            base_val = self.value[State()].val[0]
            self.resetGradient()
            self.backwardPass(advType, 1)
            self.cumulantGradient.resetGradient()
            self.accGradient()
            epsilon = 1e-07
            for s in self.policy.actionDist:
                for t in ALL_TYPES():
                    for a in s.getValidActions():
                        self.policy.actionDist[s][t].val[a] += epsilon
                        self.forwardPass(advType)
                        next_val = self.value[State()].val[0]
                        self.policy.actionDist[s][t].val[a] -= epsilon
                        derivative = (next_val - base_val) / epsilon
                        print(str(derivative) + ' ' + str(self.cumulantGradient.actionDist[s][t].grad[a]))
                        assert abs(derivative - self.cumulantGradient.actionDist[s][t].grad[a]) < 1e-05
    
    def computeSoftExpGradient(self):
        self.cumulantGradient.resetGradient()
        typeDist = initDist()
        expSum = 0
        for i in range(NUM_TYPES):
            self.resetGradient()
            self.forwardPass(TYPES[i])
            expSum += self.value[State()].val[0] * typeDist[TYPES[i]]
            self.backwardPass(TYPES[i], typeDist[TYPES[i]])
            self.accGradient()
        # Add border penalty
        for s in self.policy.actionDist:
            for t in ALL_TYPES():
                for a in s.getValidActions():
                    self.cumulantGradient.actionDist[s][t].grad[a] -= self.borderPenalty / self.policy.actionDist[s][t].val[a]
        return expSum

class LineSearch:
    def __init__(self, agentID):
        self.tol = 1e-10
        self.agentID = agentID
        self.softexp = SoftExp(agentID)
        self.policy = Policy(agentID)
        self.gradient = Policy(agentID)
    
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
    
    def getNewPolicy(self, stepSize):
        for s in self.policy.actionDist:
            for t in ALL_TYPES():
                for a in s.getValidActions():
                    self.softexp.policy.actionDist[s][t].val[a] = self.policy.actionDist[s][t].val[a] + self.gradient.actionDist[s][t].grad[a] * stepSize
                # self.correct(self.softexp.policy.actionDist[s][t].val, s.getValidActions())
                # sum = 0
                # for a in s.getValidActions():
                #     sum += self.softexp.policy.actionDist[s][t].val[a]
                #     assert self.softexp.policy.actionDist[s][t].val[a] >= 0
                # assert abs(sum - 1) < 1e-10

    def getAlignment(self):
        self.softexp.computeSoftExpGradient()
        alignment = 0
        for s in self.policy.actionDist:
            for t in ALL_TYPES():
                for a in s.getValidActions():
                    alignment += self.softexp.cumulantGradient.actionDist[s][t].grad[a] * self.gradient.actionDist[s][t].grad[a]
        return alignment

    def one_step(self):
        self.softexp.policy.copy(self.policy)
        softExp = self.softexp.computeSoftExpGradient()
        # print("EXPLOITABILITY: " + str(softExp))
        self.gradient.copy(self.softexp.cumulantGradient)
        self.getNewPolicy(-0.01)
        self.policy.copy(self.softexp.policy)
        self.softexp.softMult *= 1.01
        self.softexp.borderPenalty /= 1.01
        return softExp

    def iterate(self):
        # Find best descent direction
        self.softexp.policy.copy(self.policy)
        softExp = self.softexp.computeSoftExpGradient()
        # print("EXPLOITABILITY: " + str(softExp))
        self.gradient.copy(self.softexp.cumulantGradient)
        for s in self.policy.actionDist:
            for t in ALL_TYPES():
                sum = 0
                for a in s.getValidActions():
                    sum += self.gradient.actionDist[s][t].grad[a]
                for a in s.getValidActions():
                    self.gradient.actionDist[s][t].grad[a] = sum / len(s.getValidActions()) - self.gradient.actionDist[s][t].grad[a]
                # normGrad = []
                # for a in s.getValidActions():
                #     if self.policy.actionDist[s][t].val[a] < borderSpace and self.gradient.actionDist[s][t].grad[a] > 0:
                #         self.gradient.actionDist[s][t].grad[a] = 0
                #         print("ZEROGRAD######################" + str(s) + ' ' + str(t) + ' ' + str(a))
                #     else:
                #         normGrad.append(a)
                # sum = 0
                # for a in normGrad:
                #     sum += self.gradient.actionDist[s][t].grad[a]
                # for a in normGrad:
                #     self.gradient.actionDist[s][t].grad[a] = sum / len(normGrad) - self.gradient.actionDist[s][t].grad[a]
        # print("GRADIENT:")
        # self.gradient.printGrad()
        # Find maximum step size that stays within bounds
        maxStep = 1e+10
        for s in self.policy.actionDist:
            for t in ALL_TYPES():
                for a in s.getValidActions():
                    grad = self.gradient.actionDist[s][t].grad[a]
                    if grad < 0:
                        stepBound = self.policy.actionDist[s][t].val[a] / (-grad)
                        maxStep = min(maxStep, stepBound)
        lower = 0
        upper = maxStep
        # self.getNewPolicy(lower)
        # align_lower = self.getAlignment()
        # align_upper = None

        while upper - lower > self.tol:
            middle = (upper + lower) / 2
            # if align_lower == None or align_upper == None:
            #     middle = (upper + lower) / 2
            # else:
            #     middle = (-upper * align_lower + lower * align_upper) / (-align_lower + align_upper)
            # print(middle)
            self.getNewPolicy(middle)
            alignment = self.getAlignment()
            print(alignment)
            if alignment < 0:
                lower = middle
                # align_lower = alignment
            else:
                upper = middle
                # align_upper = alignment
        self.getNewPolicy(lower)
        self.policy.copy(self.softexp.policy)
        self.softexp.softMult *= 1.1
        self.softexp.borderPenalty /= 1.1
        # print(self.softexp.softMult)
        # print(self.softexp.borderPenalty)
        return softExp

class SoftExpGame:
    def __init__(self):
        self.policyOpt = [LineSearch(0), LineSearch(1)]
    def runOpt(self, numIter):
        start_time = time.time()
        for i in range(numIter):
            upperExp = self.policyOpt[0].iterate()
            lowerExp = self.policyOpt[1].iterate()
            print("Step " + str(i) + " EXPLOIT " + str(upperExp + lowerExp) + " Time Stamp: " + str(time.time() - start_time))
            # print(self.policyOpt[0].policy)
            # print(self.policyOpt[1].policy)

# test = SoftExp(0)
# test.test()

# test.computeSoftExpGradient()
# test.cumulantGradient.printGrad()

# test = LineSearch(0)
# for i in range(300):
#     test.iterate()
#     print("#############POLICY#############")
#     print(test.policy)

game = SoftExpGame()
game.runOpt(100)
# print(game.policyOpt[0].policy)
# print(game.policyOpt[1].policy)