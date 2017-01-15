from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# *** PASTE DATA UNDER HERE ***

# Disk Accesses 1-NNS
y1_SRTree = [14.487, 37.3868, 94.4028]
y1_E02 = [1, 2, 3]
y1_E05 = [4, 5, 6]
y1_E10 = [0.273, 0.084, 0.061]
y1_E20 = [7, 8, 9]
# Disk Accesses 10-NNS
y10_SRTree = [14.487, 37.3868, 94.4028]
y10_E02 = [1, 2, 3]
y10_E05 = [1, 2, 3]
y10_E10 = [0.273, 0.084, 0.061]
y10_E20 = [1, 2, 3]

# *** PASTE DATA OVER HERE ***



k = 100
dataMultiplier = 100000	# How much to multiple to original data
numQueryPoints = 500	# n
maxNumberOfHashTables = 10
numDataPoints = 19000

# Dimensions
x = [8, 16, 32]


# Plot figure 1-NNS
fig1 = plt.figure("Dimensions - 1-NNS")
ax = fig1.add_subplot(111)
ax.set_title("k=" + str(k) + ", 1-NNS")
ax.set_xlabel("Dimensions")
ax.set_ylabel("Disk Accesses")
ax.plot(x, y1_SRTree, 'k--s', label="SR-Tree")
ax.plot(x, y1_E02, 'r-s', label="LSH, error=.02")
ax.plot(x, y1_E05, 'g-*', label="LSH, error=.05")
ax.plot(x, y1_E10, 'b-^', label="LSH, error=.1")
ax.plot(x, y1_E20, 'm-o', label="LSH, error=.2")
ax.axis([0, x[-1] + 1, 0, max(max(y1_SRTree), max(y1_E02), max(y1_E05), max(y1_E10), max(y1_E20)) + 5])
ax.legend(loc=2)

# Plot figure 10-NNS
fig10 = plt.figure("Dimensions - 10-NNS")
ax = fig10.add_subplot(111)
ax.set_title("k=" + str(k) + ", 10-NNS")
ax.set_xlabel("Dimensions")
ax.set_ylabel("Disk Accesses")
ax.plot(x, y10_SRTree, 'k--s', label="SR-Tree")
ax.plot(x, y10_E02, 'r-s', label="LSH, error=.02")
ax.plot(x, y10_E05, 'g-*', label="LSH, error=.05")
ax.plot(x, y10_E10, 'b-^', label="LSH, error=.1")
ax.plot(x, y10_E20, 'm-o', label="LSH, error=.2")
ax.axis([0, x[-1] + 1, 0, max(max(y10_SRTree), max(y10_E02), max(y10_E05), max(y10_E10), max(y10_E20)) + 5])
ax.legend(loc=2)
plt.show()
