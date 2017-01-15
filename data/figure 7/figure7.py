from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# *** PASTE DATA UNDER HERE ***

k = 100
dataMultiplier = 100000	# How much to multiple to original data
numQueryPoints = 500	# n
maxNumberOfHashTables = 10
numDataPoints = 19000

# Dimensions
x = [0.273, 0.084, 0.061, 0.033, 0.011, 0.008, 0.007, 0.003, 0.002, 0.002]

# Disk Accesses 1-NNS
y1_E05 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y1_E10 = [0.273, 0.084, 0.061, 0.033, 0.011, 0.008, 0.007, 0.003, 0.002, 0.002]
# Disk Accesses 10-NNS
y10_E05 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y10_E10 = [0.273, 0.084, 0.061, 0.033, 0.011, 0.008, 0.007, 0.003, 0.002, 0.002]

# *** PASTE DATA OVER HERE ***

# Plot figure 1-NNS
fig1 = plt.figure("Dimensions - 1-NNS")
ax = fig1.add_subplot(111)
ax.set_title("k=" + str(k) + ", 1-NNS")
ax.set_xlabel("Dimensions")
ax.set_ylabel("Disk Accesses")
ax.plot(x, y1_E05, 'r-')
ax.plot(x, y1_E05, 'bs')
ax.plot(x, y1_E10, 'g-')
ax.plot(x, y1_E10, 'b*')
ax.axis([0, x[-1] + 1, 0, max(max(y1_E05), max(y1_E10)) + 0.1])

# Plot figure 10-NNS
fig10 = plt.figure("Dimensions - 10-NNS")
ax = fig10.add_subplot(111)
ax.set_title("k=" + str(k) + ", 10-NNS")
ax.set_xlabel("Dimensions")
ax.set_ylabel("Disk Accesses")
ax.plot(x, y10_E05, 'r-')
ax.plot(x, y10_E05, 'bs')
ax.plot(x, y10_E10, 'g-')
ax.plot(x, y10_E10, 'b*')
ax.axis([0, x[-1] + 1, 0, max(max(y10_E05), max(y10_E10)) + 0.1])
plt.show()
