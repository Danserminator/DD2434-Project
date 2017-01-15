from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# *** PASTE DATA UNDER HERE ***

k = 100
d = 32
dataMultiplier = 100000	# How much to multiple to original data
numQueryPoints = 500	# n
maxNumberOfHashTables = 10
numDataPoints = 19000
readQueryIndicesFromFile = True
K = 1

# Number of hash tables
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Error
y = [0.273, 0.084, 0.061, 0.033, 0.011, 0.008, 0.007, 0.003, 0.002, 0.002]

# *** PASTE DATA OVER HERE ***

# Plot figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("n=" + str(numDataPoints) + ", d=" + str(d) + ", k=" + str(k))
ax.set_xlabel("Number of indices")
ax.set_ylabel("Error")
ax.plot(x, y, 'r-')
ax.plot(x, y, 'bs')
ax.axis([0, x[-1] + 1, 0, max(y) + 0.1])
plt.show()
