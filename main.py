
from PSRO import *

a = PSRO()

# RUN PSRO

# fileName = "dicesum.pickle"
# fileName = "liarsdice.pickle"
fileName = "liarsdice_wild.pickle"

# DICE SUM PARAMETERS
# LR = 0.02
# Iterations = 1000
# gives exploitability 0.097

# LIARS DICE PARAMETERS
# LR = 0.05
# Iterations = 10000
# gives exploitability 0.00199

# WILD LIARS DICE PARAMETERS
# LR = 0.05
# Iterations = 1000
# gives exploitability 0.099

# RUN PSRO
# with open(fileName, 'rb') as f:
#     a.policies = pickle.load(f)
# a.runPSRO(1000)
# with open(fileName, 'wb') as f:
#     pickle.dump(a.returnPolicies, f, pickle.HIGHEST_PROTOCOL)

# LOAD SAVED POLICY
with open(fileName, 'rb') as f:
    a.returnPolicies = pickle.load(f)

a.policies[0] = a.returnPolicies[0].copy()
a.policies[1] = a.returnPolicies[1].copy()

s = State()
for t in ALL_TYPES():
    print("Type " + str(t) + " Policy: " + str(a.returnPolicies[0].actionDist[s][t]))

for act in s.getValidActions():
    new_s = s.copy()
    new_s.makeAction(act)
    print("____ACTION " + str(act) + "_____\n")
    for t in ALL_TYPES():
        print("Type " + str(t) + " Policy: " + str(a.returnPolicies[1].actionDist[new_s][t]))

# a.runPSRO(1)

sum = 0
count = 1
while True:
    print("__________NEW GAME:____________")
    score = a.play(False)
    sum += score
    print("Cumulative score: " + str(sum) + "/" + str(count))
    count += 1