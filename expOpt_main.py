
from expOpt import *

a = ExpOpt()


# print(a.policies[0])
# print(a.policies[1])

# RUN Exploitability Optimization

# with open("liarsdice_wild.pickle", 'rb') as f:
#     a = pickle.load(f)
# a.play()

a.runOpt(1000, 0.8, 0.99)
with open("liarsdice_wild.pickle", 'wb') as f:
    pickle.dump(a, f, pickle.HIGHEST_PROTOCOL)
