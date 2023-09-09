
from expOpt import *

a = ExpOpt()


# RUN Exploitability Optimization

# with open(fileName, 'rb') as f:
#     a = pickle.load(f)

a.runExpOut(1000)
with open(fileName, 'wb') as f:
    pickle.dump(a, f, pickle.HIGHEST_PROTOCOL)
