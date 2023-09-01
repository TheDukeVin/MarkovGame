
from race import *
from nash import *
import matplotlib.pyplot as plt
from matplotlib import colors

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

def plot(T):
    maxSize = 40

    data = np.zeros(shape=(maxSize, maxSize))

    for i in range(maxSize):
        for j in range(maxSize):
            data[i][j] = -1
            s = State()
            s.time = T
            s.val = [i, j]
            if s in a.store:
                p1, _, _ = a.store[s]
                data[i][j] = p1[1]

    # create discrete colormap
    cmap = colors.ListedColormap(['white', 'red', 'lightcoral', 'grey', 'black'])
    bounds = [-1, -0.01, 0.01, 0.5, 0.99, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(-.5, maxSize, 1));
    ax.set_yticks(np.arange(-.5, maxSize, 1));

    plt.savefig("policy" + str(T) + ".png")

for i in range(State.TIME_HORIZON):
    plot(i)