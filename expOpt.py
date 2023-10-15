
import pickle
import time

from nash import *

# from high import *
# from dicesum import *
from liarsdice import *

"""
rsync -r MarkovGame kevindu@login.rc.fas.harvard.edu:./MultiagentSnake
rsync -r kevindu@login.rc.fas.harvard.edu:./MultiagentSnake/LSTM/net.out MarkovGame
"""

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

def sign(x):
    if x < 0: return -1
    return 1

class ExpOpt:
    # FOR TWO PLAYERS ONLY
    def __init__(self):
        self.policies = [Policy(0), Policy(1)]
        self.responsePolicies = [Policy(0), Policy(1)]
        self.gradientPolicies = [Policy(0), Policy(1)]
        self.gradient = [Policy(0), Policy(1)]
        self.momentum = [Policy(0), Policy(1)]
        self.N = 0
        self.learnRate = 1
        self.momentumRate = 0
    
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
    
    def incPolicy(self, agentID):
        for s in self.policies[agentID].actionDist:
            for t in ALL_TYPES():
                # sum = 0
                # for a in s.getValidActions():
                #     sum += self.gradient[agentID].actionDist[s][t][a]
                # for a in s.getValidActions():
                #     self.gradient[agentID].actionDist[s][t][a] -= sum / len(s.getValidActions())
                #     val = self.gradient[agentID].actionDist[s][t][a]
                #     if sign(self.momentum[agentID].actionDist[s][t][a]) == sign(self.gradient[agentID].actionDist[s][t][a]):
                #         val += self.momentum[agentID].actionDist[s][t][a]
                #     else:
                #         self.momentum[agentID].actionDist[s][t][a] = 0
                #     self.momentum[agentID].actionDist[s][t][a] += self.gradient[agentID].actionDist[s][t][a]
                #     self.policies[agentID].actionDist[s][t][a] += val * exploit
                for a in s.getValidActions():
                    self.momentum[agentID].actionDist[s][t][a] *= self.momentumRate
                    self.momentum[agentID].actionDist[s][t][a] += self.gradient[agentID].actionDist[s][t][a]
                    self.policies[agentID].actionDist[s][t][a] += self.momentum[agentID].actionDist[s][t][a] # * self.learnRate
                self.correct(self.policies[agentID].actionDist[s][t], s.getValidActions())

    # def getAlignment(self, policy, grad, step, initGrad, agentID):
    #     self.responsePolicies[agentID].copy(policy)
    #     for s in grad.actionDist:
    #         for t in ALL_TYPES():
    #             for a in s.getValidActions():
    #                 self.responsePolicies[agentID].actionDist[s][t][a] += grad.actionDist[s][t][a] * step
    #     # print(self.responsePolicies[agentID])
    #     exp = self.bestResponse(1-agentID)
    #     self.gradientPolicies[1-agentID].copy(self.responsePolicies[1-agentID])
    #     self.gradientPolicies[agentID].copy(self.responsePolicies[agentID])
    #     self.getGradient(agentID)
    #     # print("NEW GRADIENT: ")
    #     # print(self.gradient[agentID])
    #     alignment = 0
    #     for s in initGrad.actionDist:
    #         for t in ALL_TYPES():
    #             for a in s.getValidActions():
    #                 alignment += initGrad.actionDist[s][t][a] * self.gradient[agentID].actionDist[s][t][a]
    #     # print("ALIGNMENT: " + str(alignment))
    #     return alignment


    # def lineSearch(self, agentID): # Returns whether the next gradient vector was found
    #     # normalize each gradient so it sums to 0
    #     tol = 0.01 * np.exp(-self.N*0.1)
    #     initGrad = Policy(agentID)
    #     initGrad.copy(self.gradient[agentID])
    #     for s in initGrad.actionDist:
    #         for t in ALL_TYPES():
    #             gradActions = []
    #             for a in s.getValidActions():
    #                 if self.policies[agentID].actionDist[s][t][a] < tol and initGrad.actionDist[s][t][a] < 0:
    #                     initGrad.actionDist[s][t][a] = 0
    #                 else:
    #                     gradActions.append(a)
    #             sum = 0
    #             for a in gradActions:
    #                 sum += initGrad.actionDist[s][t][a]
    #             for a in gradActions:
    #                 initGrad.actionDist[s][t][a] -= sum / len(gradActions)
    #     # print("POLICY:")
    #     # print(self.policies[agentID])
    #     # print("GRADIENT:")
    #     # print(initGrad)
    #     maxStep = 1e+10
    #     for s in initGrad.actionDist:
    #         for t in ALL_TYPES():
    #             for a in s.getValidActions():
    #                 grad = initGrad.actionDist[s][t][a]
    #                 if abs(grad) < tol:
    #                     continue
    #                 if grad > 0:
    #                     stepBound = (1 - self.policies[agentID].actionDist[s][t][a]) / grad
    #                 else:
    #                     stepBound = self.policies[agentID].actionDist[s][t][a] / (-grad)
    #                 maxStep = min(maxStep, stepBound)
    #     lower = 0
    #     upper = maxStep
    #     while upper - lower > tol:
    #         middle = (upper + lower) / 2
    #         alignment = self.getAlignment(self.policies[agentID], initGrad, middle, initGrad, agentID)
    #         if alignment > 0:
    #             lower = middle
    #         else:
    #             upper = middle
    #     alignment_lower = self.getAlignment(self.policies[agentID], initGrad, lower, initGrad, agentID)
    #     lower_gradient = Policy(agentID)
    #     lower_gradient.copy(self.gradient[agentID])
    #     alignment_upper = self.getAlignment(self.policies[agentID], initGrad, upper, initGrad, agentID)
    #     upper_gradient = Policy(agentID)
    #     upper_gradient.copy(self.gradient[agentID])

    #     for s in initGrad.actionDist:
    #         for t in ALL_TYPES():
    #             for a in s.getValidActions():
    #                 self.gradient[agentID].actionDist[s][t][a] = (
    #                     lower_gradient.actionDist[s][t][a] * (-alignment_upper) + 
    #                     upper_gradient.actionDist[s][t][a] * alignment_lower
    #                 )
    #     # print("NEW GRADIENT")
    #     # print(self.gradient[agentID])
    #     # print("NEW POLICY")
    #     # print(self.policies[agentID])
    #     # print("ALIGNMENTS")
    #     # print(alignment_lower, alignment_upper)

    #     self.policies[agentID].copy(self.responsePolicies[agentID])
    #     return abs(alignment_lower) + abs(alignment_upper) > tol

    def iterate(self):
        exploitability = 0
        for agentID in range(2):
            self.responsePolicies[agentID].copy(self.policies[agentID])
            exploitability += self.bestResponse(1-agentID)
            self.gradientPolicies[agentID].copy(self.policies[agentID])
            self.gradientPolicies[1-agentID].copy(self.responsePolicies[1-agentID])
            self.getGradient(agentID)
            self.incPolicy(agentID)

            
            # if True:
            # # if not self.foundGradient[agentID]:
            #     # print("GRADIENT NOT FOUND___________________________________")
            # # if self.N == 0:
            #     self.gradientPolicies[agentID].copy(self.policies[agentID])
            #     self.gradientPolicies[1-agentID].copy(self.responsePolicies[1-agentID])
            #     self.getGradient(agentID)
            # self.foundGradient[agentID] = self.lineSearch(agentID)
        self.N += 1
        return exploitability
    
    def runOpt(self, numIter, momentum, lrAnneal):
        start_time = time.time()
        # self.foundGradient = [False, False]
        self.momentum[0].resetGradient()
        self.momentum[1].resetGradient()
        evalPeriod = 10
        sumExp = 0
        lastAvg = 1e+07
        self.momentumRate = momentum
        log = []
        for i in range(numIter):
            self.learnRate *= lrAnneal
            # if i % evalPeriod == 0 and i > 0:
            #     if sumExp / evalPeriod > lastAvg:
            #         self.learnRate *= 0.1
            #     lastAvg = sumExp / evalPeriod
            #     sumExp = 0
            #     print("AVG Exploitability: " + str(lastAvg))
            self.exploitability = self.iterate()
            # sumExp += self.exploitability
            log.append(str(self.exploitability))
            print("Step: " + str(self.N) + " Exploitability: " + str(self.exploitability) + " Learn Rate: " + str(self.learnRate) + " Time Stamp: " + str(time.time() - start_time))
            # if self.exploitability < 1e-07:
            #     break
        with open("log.out", 'w') as f:
            f.write(','.join(log))
    
    def bestResponse(self, agentID):
        sum = 0
        for t, p in initDist().items():
            val = self.bestResponseRec(agentID, State(), t, oppDist(t))
            sum += p * val
        return sum
    
    def bestResponseRec(self, agentID, s : State, t : Type, oppType : dict):
        # Reads other policies from self.responsePolicies
        # Calculates best response, updates self.responsePolicies[agentID]
        # Returns value of best response

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
                candVal = self.bestResponseRec(agentID, new_s, t, oppType)
                if maxVal < candVal:
                    maxVal = candVal
                    bestAction = a
                self.responsePolicies[agentID].actionDist[s][t][a] = 0
            self.responsePolicies[agentID].actionDist[s][t][bestAction] = 1
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
                newOppType[ot] = oppType[ot] * self.responsePolicies[1-agentID].actionDist[s][ot][a]
                bayesScale += newOppType[ot]
            if bayesScale < 1e-07:
                continue
            for ot in oppType:
                newOppType[ot] /= bayesScale
            sum += bayesScale * self.bestResponseRec(agentID, new_s, t, newOppType)
        return sum
    
    def gradientTest(self, agentID):
        epsilon = 1e-04
        errorSum = 0
        val = self.getGradient(agentID)
        for s in self.gradientPolicies[agentID].actionDist:
            for t in ALL_TYPES():
                for a in s.getValidActions():
                    self.gradientPolicies[agentID].actionDist[s][t][a] += epsilon
                    neigh_val = self.getGradient(agentID)
                    self.gradientPolicies[agentID].actionDist[s][t][a] -= epsilon
                    # print(neigh_val - val, epsilon * self.gradient[agentID].actionDist[s][t][a])
                    errorSum += abs((neigh_val - val) - epsilon * self.gradient[agentID].actionDist[s][t][a])
        assert(errorSum < 1e-10)

    def getGradient(self, agentID):
        # Calculates gradient of EV wrt each action given self.gradientPolicies
        # Calculates value of state using current policies
        self.gradient[agentID].resetGradient()
        val = 0
        for t, p in initDist().items():
            for ot, op in oppDist(t).items():
                types = [None, None]
                types[agentID] = t
                types[1-agentID] = ot
                val += p * op * self.accGradientRec(agentID, State(), types, p * op)
        return val
    
    def accGradientRec(self, agentID, s : State, types : list[Type], prob : float):
        if s.endState:
            return s.getEndValue(types)[agentID]
        sum = 0
        for a in s.getValidActions():
            new_s = s.copy()
            new_s.makeAction(a)
            action_prob = self.gradientPolicies[s.currPlayer].actionDist[s][types[s.currPlayer]][a]
            val = self.accGradientRec(agentID, new_s, types, prob * action_prob)
            sum += action_prob * val
            if s.currPlayer == agentID:
                pivot = 1e-03
                mult = min([self.learnRate, prob, self.learnRate * prob**(np.log(pivot/self.learnRate)/np.log(pivot))])
                self.gradient[agentID].actionDist[s][types[agentID]][a] += val * mult #prob
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
                    print("Opponent policy: " + str(self.policies[0].actionDist[s][t0]))
                a = np.random.choice(State.NUM_ACTION, p=self.policies[0].actionDist[s][t0])
            else:
                if reveal:
                    print("Your policy: " + str(self.policies[1].actionDist[s][t1]))
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