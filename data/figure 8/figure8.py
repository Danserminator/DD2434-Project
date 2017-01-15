from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# *** PASTE DATA UNDER HERE ***

# Disk Accesses
y_E05 = [1, 2, 3, 4, 5]
y_E20 = [1, 2, 3, 4, 5]
y_E10 = [1, 2, 3, 4, 5]

# *** PASTE DATA OVER HERE ***




k = 100
d = 32
dataMultiplier = 100000	# How much to multiple to original data
numQueryPoints = 500	# n
maxNumberOfHashTables = 10
numDataPoints = 19000

# Number of nearest neighbors
x = [1, 10, 20, 50, 100]

# Plot figure 1-NNS
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("n=" + str(numDataPoints) + ", d=" + str(d) + ", k=" + str(k) + ", 1-NNS")
ax.set_xlabel("Number of nearest neighbors")
ax.set_ylabel("Disk Accesses")
ax.plot(x, y_E05, 'k-*', label="Error=.05")
ax.plot(x, y_E10, 'r-s', label="Error=.1")
ax.plot(x, y_E20, 'b-^', label="Error=.2")
ax.axis([0, x[-1] + 1, 0, max(max(y_E05), max(y_E10), max(y_E20)) + 0.1])
ax.legend(loc=2)
plt.show()
