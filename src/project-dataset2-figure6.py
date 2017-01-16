from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
from lsh import *
import time
import os
from sklearn.neighbors import NearestNeighbors
import sys
import pickle	# Writing and reading files
import pandas as pd

# dataSet = "ColorHistogram.asc"
dataSet = "tabledata.dat"
projectPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# All the variables basically...
k = 65            # How many bits to use for hashing
# k = 100
# k = 700

# B = 100							# Maximum number of points in a bucket
# B = 300							# Maximum number of points in a bucket
B = 25000							# Maximum number of points in a bucket
numQueryPoints = 500			# Number of query points
# numQueryPoints = 1000			# Number of query points
readQueryIndicesFromFile = True
maxNumberOfHashTables = 10		# Number of hash tables to use

# numDataPoints = 1000			# sys.maxint for all
# numDataPoints = 2000			# sys.maxint for all
# numDataPoints = 5000			# sys.maxint for all
# numDataPoints = 10000			# sys.maxint for all
# numDataPoints = 19000			# sys.maxint for all
# numDataPoints = 27000			# sys.maxint for all
numDataPoints = sys.maxint			# sys.maxint for all

# numDataPoints = sys.maxint
# dataMultiplier = 10000			# How much to multiple to original data
dataMultiplier = 100000			# How much to multiple to original data

# K = 1							# Number of neighbours to find
K = 10							# Number of neighbours to find
# d = 32							# Number of dimensions (sys.maxint for all)
d = sys.maxint				# Number of dimensions (sys.maxint for all)
tries = 5						# How many times it should be done


def getDistanceL2(p1, p2):
	return np.sqrt(np.sum(np.power(np.abs(p1 - p2), 2)))

def getDistanceL1(p1, p2):
	return np.sum(np.abs(p1 - p2))

def getApproxNN(P, T, point, K):
	S = set()
	for t in T:
		S = S.union(t.get(point))
		
	Ss = np.array([P[x] for x in S])
	
	return getExactNNB(Ss, point, K)

# Is this allowed?!
def getExactNNB(P, point, K, nbrs = None):
	K = np.minimum(K, len(P))
	if K > 0:
		if nbrs == None:
			nbrs = NearestNeighbors(n_neighbors = K, algorithm = 'ball_tree').fit(P)
		distances, indices = nbrs.kneighbors([point])
		# Return distance instead of point since multiple points can have the same distance
		return [tuple(x) for x in distances][0]
		#return tuple(tuple(P[x].tolist()[0]) for x in indices)
	
	return []

def getExactNNB2(P, point, K):
	K = np.minimum(K, len(P))
	if K > 0:
		d = list(map((lambda p: (getDistanceL2(p, point), tuple(p.tolist()))), P))
		dd = sorted(d, key=lambda dist: dist[0])
		print(zip(*dd[:K])[1])
		return zip(*dd[:K])[1]
	return []
	
def getExactNN(P, point, K):
	KNN = []
	distances = []
	for neighbour in P:
		distance = getDistanceL2(point, neighbour)
		if len(KNN) < K:
			KNN.append(tuple(neighbour.tolist()))
			distances.append(distance)
		else:
			maxDist = distance
			maxDistIndex = -1
			for i in range(0, len(distances)):
				if distances[i] > maxDist:
					maxDist = distances[i]
					maxDistIndex = i
			if maxDistIndex != -1:
				KNN[maxDistIndex] = tuple(neighbour.tolist())
				distances[maxDistIndex] = distance
			
	# Sort so nearest first	
	temp = zip(distances, KNN)
	temp.sort()
	
	return [x for y, x in temp]

def figure4(P, k, B):
	print("Building tree for exact K-NN")
	start = time.time()
	# Build the tree for exact K-NN here so we only have to do it ones
	nbrs = NearestNeighbors(n_neighbors = K, algorithm = 'ball_tree').fit(P)
	print("Done building the tree, it took: " + str(time.time() - start) + " seconds")
	
	error = [0.0] * maxNumberOfHashTables
	missRatio = [0.0] * maxNumberOfHashTables
	indices = range(1, maxNumberOfHashTables + 1)
	for i in range(0, tries):
		print("Start preprocessing of the data for the " + str(i+1) + " time")
		start = time.time()
		T = lsh.preprocessing(P, maxNumberOfHashTables)
		print("Preprocessing done, created " + str(maxNumberOfHashTables) + " hash tables in " + str(time.time() - start) + " seconds")
		
		for l in indices:
			start = time.time()
			error_l = [0.0] * K
			numMisses = 0
			#for test in range(0, tries):
			for queryPoint in queryPoints:
				approx = getApproxNN(P, T[:l], np.array(queryPoint), K)
				exact = getExactNNB(P, np.array(queryPoint), K, nbrs)
		
				for idx, val in enumerate(approx):
					if exact[idx] > 0 :
						error_l[idx] += (val / float(exact[idx])) - 1	# They forgot to write -1?
			
				if len(approx) < K:
					numMisses += K - len(approx)
	
			error_l = sum(error_l) / float(len(error_l))
			error_l /= float(numQueryPoints)
			error[l-1] += error_l
			
			missRatio[l-1] += numMisses / (float(numQueryPoints) * K)
			
			print("Time taken when using " + str(l) + " hash tables: " + str(time.time() - start) + " seconds")
			
	for i in range(0, len(error)):
		error[i] = round(error[i] / float(tries), 2)
		missRatio[i] = round(missRatio[i] / float(tries), 3)

	# Write to file
	with open(projectPath + "/data/figure4.txt", "a") as f:
		f.write("--------------------\n")
		f.write("k = " + str(k) + "\n")
		f.write("d = " + str(P.shape[-1]) + "\n")
		f.write("dataMultiplier = " + str(dataMultiplier) + "\t# How much to multiple to original data\n")
		f.write("numQueryPoints = " + str(numQueryPoints) + "\t# n\n")
		f.write("maxNumberOfHashTables = " + str(maxNumberOfHashTables) + "\n")
		f.write("numDataPoints = " + str(numDataPoints) + "\n")
		f.write("readQueryIndicesFromFile = " + str(readQueryIndicesFromFile) + "\n")
		f.write("K = " + str(K) + "\n\n")
		
		f.write("# Number of hash tables\n")
		f.write("x = [")
		for i, v in enumerate(indices):
			f.write(str(v))
			if i != len(indices) - 1:
				f.write(", ")
		f.write("]\n\n")
		
		f.write("# Error\n")
		f.write("y = [")
		for i, v in enumerate(error):
			f.write(str(v))
			if i != len(error) - 1:
				f.write(", ")
		f.write("]\n\n")
		
		f.write("# Miss ratio\n")
		f.write("Miss ratio = [")
		for i, v in enumerate(missRatio):
			f.write(str(v))
			if i != len(missRatio) - 1:
				f.write(", ")
		f.write("]\n\n")
		
	# Plot figure
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title("n=" + str(len(P)) + ", d=" + str(P.shape[-1]) + ", k=" + str(k))
	ax.set_xlabel("Number of indices")
	ax.set_ylabel("Error")
	ax.plot(indices, error, 'r-')
	ax.plot(indices, error, 'bs')
	ax.axis([0, indices[-1] + 1, 0, max(error) + 0.1])
	plt.show()

def figure6():
	pass
		
def output():
	# Just for output
	for i in range(0, len(T)):
		print("Bucket " + str(i+1))
		print("\tBucket\t:\tPoints")
		for key in T[i].bucket:
			print("\t" + str(key) + "\t:\t" + str(T[i].bucket[key][0]), end="")
			for j in range(1, len(T[i].bucket[key])):
				print(", " + str(T[i].bucket[key][j]), end="")
			print("")
		'''
		checkPoint = [90, 80]
		print("Point " + str(checkPoint) + " is hopefully similar to:")
		similarPoints = T[i].get(np.array(checkPoint))
		if len(similarPoints) == 0:
			print("\tNone")
		else:
			print("\t" + str(similarPoints[0]), end="")
			for j in range(1, len(similarPoints)):
				print(", " + str(similarPoints[j]), end="")
			print("")
		'''
		print("\n")

	K = 1
	queryPoint = np.array([70, 9])
	print("Get " + str(K) + " approximate nearest neighbors to " + str(queryPoint) + ":")
	print("\t" + str(getApproxNN(T, np.array(queryPoint), K)))
	print("Get " + str(K) + " nearest neighbors to " + str(queryPoint) + ":")
	print("\t" + str(getExactNN(P, np.array(queryPoint), K)))
	
# Open and read database into P




# P2 = df.as_matrix() * 1000000
# P2 = df.as_matrix()

# print(P2.__class__)
# print(P2.shape)

# print(P2[0:5, 0:5])
# print(P2.head())
# P = np.delete(np.genfromtxt(projectPath + "/data/" + dataSet,delimiter=" ") * dataMultiplier,0,1).astype(int)
datasetFullPath = projectPath + "/data/" + dataSet
df = pd.read_csv(datasetFullPath)
df = df.ix[:, 2:62]
P = df.as_matrix() * dataMultiplier


lsh = LSH(k, B)
	
queryPoints = [None] * numQueryPoints

if readQueryIndicesFromFile:
	with open(projectPath + "/data/queryIndices.txt", "rb") as f:
		randNums = pickle.load(f)[:numQueryPoints]
else:
	randNums = np.random.choice(len(P), numQueryPoints, replace=False)
	# Write these to file
	with open(projectPath + "/data/queryIndices.txt", "wb") as f:
		pickle.dump(randNums, f)
	
if numQueryPoints > len(randNums):
	print("Not enough points for the number of query points specified")
	sys.exit(1)

for idx, randNum in enumerate(randNums):
	queryPoints[idx] = P[randNum][:d]

P = np.delete(P, randNums, 0)[:numDataPoints, :d]
print("Size of data set: " + str(len(P)))
print("Size of test set: " + str(len(queryPoints)))

#output()
figure4(P, k, B)
