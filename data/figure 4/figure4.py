from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# *** PASTE DATA UNDER HERE ***

# Error
y = [0.43, 0.2, 0.13, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02]

# *** PASTE DATA OVER HERE ***



k = 65
d = 32
dataMultiplier = 100000	# How much to multiple to original data
numQueryPoints = 500	# n
maxNumberOfHashTables = 10
numDataPoints = 19000
readQueryIndicesFromFile = True
K = 1

# Number of hash tables
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Plot figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("n=" + str(numDataPoints) + ", d=" + str(d) + ", k=" + str(k))
ax.set_xlabel("Number of indices")
ax.set_ylabel("Error")
ax.plot(x, y, 'k-s')
ax.axis([0, x[-1] + 1, 0, max(y) + 0.1])
plt.show()
