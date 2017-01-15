import numpy as np
import math
import matplotlib.pyplot as plt
import time
import os

dataSet = "ColorHistogram.asc"
projectPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
numDataPoints = 5000
def getDistanceL2(p1, p2):
	return np.sqrt(np.sum(np.power(np.abs(p1 - p2), 2)))

def getExactNNB2(P, point):
	d = np.asarray((map((lambda p: getDistanceL2(p, point)), P)))
	return d




P = np.delete(np.genfromtxt(projectPath + "/data/" + dataSet,delimiter=" ") * 100000,0,1).astype(int)
randNums = np.random.choice(len(P), 0, replace=False)
P = np.delete(P, randNums, 0)[:numDataPoints]


z = len(P)
print(z)
ds = np.zeros(0)

for idx, x in enumerate(P):
	if idx < z - 1:
		d = getExactNNB2(P[-z+idx+1:],x)
		print(idx)
		ds = np.append(ds,d)

ds = ds.flatten()
print("saving")
np.savetxt('dist.txt',ds)

plt.hist(ds, bins='auto')  # plt.hist passes it's arguments to np.histogram
plt.title("Interpoint distance")
plt.show()
