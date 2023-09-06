
from race import *
from nash import *
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl

class MG:
    def __init__(self):
        self.store = {}
        self.solve(State())

    def solve(self, s):
        if s in self.store:
            return
        if s.endState:
            self.store[s] = None, None, s.getEndValue()
            return
        matrix = np.zeros((State.NUM_ACTION, State.NUM_ACTION))
        for i in range(State.NUM_ACTION):
            for j in range(State.NUM_ACTION):
                expectedValue = 0
                for new_s, prob in s.makeAction([i, j]):
                    self.solve(new_s)
                    _, _, val = self.store[new_s]
                    expectedValue += val * prob
                matrix[i][j] = expectedValue
        self.store[s] = getNash(matrix)
    
    def play(self):
        s = State()
        while not s.endState:
            p1, p2, v = self.store[s]
            print("State: " + str(s))
            print(self.store[s])
            a = np.random.choice(State.NUM_ACTION, p=p1)
            while True:
                try:
                    b = int(input("Enter Action: "))
                    if 0 <= b and b < State.NUM_ACTION:
                        break
                except ValueError:
                    pass
                print("Please enter valid action, i.e. 0 to " + str(State.NUM_ACTION-1))
            next = s.makeAction([a, b])
            index = np.random.choice(len(next), p=[n[1] for n in next])
            s = next[index][0].copy()
        print("Final State: " + str(s))
        print("Final value: " + str(s.getEndValue()))

a = MG()

# s = State()
# s.time = 1
# s.val = [10, 0]
# print(a.store[s])
# s2 = State()
# s2.time = 1
# s2.val = [10, 100]
# print(a.store[s2])

# while True:
#     a.play()

maxSize = 50

def plot_policy():
    data = np.zeros(shape=(State.TIME_HORIZON, 2*maxSize))

    for i in range(State.TIME_HORIZON):
        for j in range(2*maxSize):
            data[i][j] = -1
            s = State()
            s.time = i
            s.val = j-maxSize
            if s in a.store:
                _, p1, v = a.store[s]
                data[i][j] = p1[1]
                if abs(v-1) < 1e-07 or abs(v+1) < 1e-07:
                    data[i][j] = 2
    
    # create discrete colormap
    cmap = colors.ListedColormap(['white', 'red', 'lightcoral', 'grey', 'black', 'blue'])
    bounds = [-1, -0.001, 0.001, 0.5, 0.999, 1.001, 2]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(6,6))

    ax.imshow(data, cmap=cmap, norm=norm, interpolation='none', extent=[-maxSize*10,maxSize*10,10,0])
    ax.set_aspect(40) # you may also use am.imshow(..., aspect="auto") to restore the aspect ratio

    plt.savefig("policy.png")

def plot_value():
    data = np.zeros(shape=(State.TIME_HORIZON, 2*maxSize))

    for i in range(State.TIME_HORIZON):
        for j in range(2*maxSize):
            data[i][j] = -1
            s = State()
            s.time = i
            s.val = j-maxSize
            if s in a.store:
                _, _, v = a.store[s]
                data[i][j] = (v+1)/2

    cmap = mpl.colormaps['magma']

    fig, ax = plt.subplots(figsize=(6,6))

    ax.imshow(data, cmap=cmap, interpolation='none', extent=[-maxSize*10,maxSize*10,10,0])
    ax.set_aspect(40)

    plt.savefig("value.png")

plot_policy()
plot_value()