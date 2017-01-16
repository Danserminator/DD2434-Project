from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# *** PASTE DATA UNDER HERE ***

# Miss ratio 1-NNS k=300
# y1_E05 = [0.134, 0.166, 0.162, 0.013, 0.001, 0.0001]
# y1_E10 = [0.676, 0.255, 0.241, 0.028, 0.006, 0.002]
y1_E05 = [0.166, 0.162, 0.013, 0.001, 0.0001]
y1_E10 = [0.255, 0.241, 0.028, 0.006, 0.002]
# Miss ratio 10-NNS k=300
# y10_E05 = [0.853, 0.183, 0.172, 0.027, 0.001, 0.001]
# y10_E10 = [0.335, 0.307, 0.216, 0.040, 0.004, 0.004]
y10_E05 = [0.183, 0.172, 0.027, 0.001, 0.001]
y10_E10 = [0.307, 0.216, 0.040, 0.004, 0.004]



# *** PASTE DATA OVER HERE ***



k = 65
d = 60
dataMultiplier = 100000	# How much to multiple to original data
numQueryPoints = 500	# n
maxNumberOfHashTables = 10
# numDataPoints = 19000
numDataPoints = 27000

# Number of database points
# x = [1000, 2000, 5000, 10000, 19000, 27000]
x = [2000, 5000, 10000, 19000, 27000]

# Plot figure 1-NNS
fig1 = plt.figure("Miss Ratio - 1-NNS")
ax = fig1.add_subplot(111)
ax.set_title("d=" + str(d) + ", k=" + str(k) + ", 1-NNS")
ax.set_xlabel("Number of database points")
ax.set_ylabel("Miss ratio")
ax.plot(x, y1_E05, 'k-*', label="Error=.05")
ax.plot(x, y1_E10, 'r-s', label="Error=.1")
ax.axis([0, x[-1] + 1, 0, max(max(y1_E05), max(y1_E10)) + 0.1])
ax.legend(loc=1)

# Plot figure 10-NNS
fig10 = plt.figure("Miss Ratio - 10-NNS")
ax = fig10.add_subplot(111)
ax.set_title("d=" + str(d) + ", k=" + str(k) + ", 10-NNS")
ax.set_xlabel("Number of database points")
ax.set_ylabel("Miss ratio")
ax.plot(x, y10_E05, 'k-*', label="Error=.05")
ax.plot(x, y10_E10, 'r-s', label="Error=.1")
ax.axis([0, x[-1] + 100, 0, max(max(y10_E05), max(y10_E10)) + 0.1])
ax.legend(loc=1)
plt.show()
