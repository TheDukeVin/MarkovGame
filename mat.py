
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(loc=100, scale=10, size=(500,1,32))
hist = np.ones((32, 20)) # initialise hist
for z in range(32):
    hist[z], edges = np.histogram(data[:, 0, z], bins=np.arange(80, 122, 2))

fig, ax = plt.subplots(figsize=(6,6))

ax.imshow(hist, cmap=plt.cm.Reds, interpolation='none', extent=[10,120,32,0])
ax.set_aspect(2) # you may also use am.imshow(..., aspect="auto") to restore the aspect ratio

plt.savefig("test.png")