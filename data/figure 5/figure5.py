from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# *** PASTE DATA UNDER HERE ***

# Disk accesses 1-NNS
y1_SRTree = [6.53106, 11.2064, 19.1603, 31.998, 52.2124] #, 96.5531]
y1_E02 = [1, 2, 3, 4, 5]
y1_E05 = [1, 2, 3, 4, 5]
y1_E10 = [0.273, 0.084, 0.061, 0.033, 0.011]
y1_E20 = [0.273, 0.084, 0.061, 0.033, 0.011]
# Disk accesses 10-NNS
y10_SRTree = [8.28858, 14.2164, 24.7756, 41.7876, 68.6433] #, 133.78]
y10_E02 = [1, 2, 3, 4, 5]
y10_E05 = [1, 2, 3, 4, 5]
y10_E10 = [0.273, 0.084, 0.061, 0.033, 0.011]
y10_E20 = [0.273, 0.084, 0.061, 0.033, 0.011]

# *** PASTE DATA OVER HERE ***



k = 100
d = 32
dataMultiplier = 100000	# How much to multiple to original data
numQueryPoints = 500	# n
maxNumberOfHashTables = 10
numDataPoints = 19000

# Number of database points
x = [1000, 2000, 5000, 10000, 19000]

# Plot figure 1-NNS
fig1 = plt.figure("Miss Ratio - 1-NNS")
ax = fig1.add_subplot(111)
ax.set_title("d=" + str(d) + ", k=" + str(k) + ", 1-NNS")
ax.set_xlabel("Number of database points")
ax.set_ylabel("Disk accesses")
ax.plot(x, y1_SRTree, 'k--s', label="SR-Tree")
ax.plot(x, y1_E02, 'r-s', label="LSH, error=.02")
ax.plot(x, y1_E05, 'g-*', label="LSH, error=.05")
ax.plot(x, y1_E10, 'b-^', label="LSH, error=.1")
ax.plot(x, y1_E20, 'm-o', label="LSH, error=.2")
ax.axis([0, x[-1] + 1, 0, max(max(y1_SRTree), max(y1_E02), max(y1_E05), max(y1_E10), max(y1_E20)) + 5])
ax.legend(loc=2)

# Plot figure 10-NNS
fig10 = plt.figure("Miss Ratio - 10-NNS")
ax = fig10.add_subplot(111)
ax.set_title("d=" + str(d) + ", k=" + str(k) + ", 10-NNS")
ax.set_xlabel("Number of database points")
ax.set_ylabel("Disk accesses")
ax.plot(x, y10_SRTree, 'k--s', label="SR-Tree")
ax.plot(x, y10_E02, 'r-s', label="LSH, error=.02")
ax.plot(x, y10_E05, 'g-*', label="LSH, error=.05")
ax.plot(x, y10_E10, 'b-^', label="LSH, error=.1")
ax.plot(x, y10_E20, 'm-o', label="LSH, error=.2")
ax.axis([0, x[-1] + 1, 0, max(max(y10_SRTree), max(y10_E02), max(y10_E05), max(y10_E10), max(y10_E20)) + 5])
ax.legend(loc=2)
plt.show()
