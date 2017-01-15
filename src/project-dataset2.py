from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
from lsh import *
import time
import os
from sklearn.neighbors import NearestNeighbors
import sys

dataSet = "ColorHistogram.asc"
dataSet2 =  "tabledata.dat"
projectPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

k = 100
B = 25000
numQueryPoints = 1000
maxNumberOfHashTables = 10
numDataPoints = sys.maxint

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

def figure4(P, k, B, T, queryPoints, error):
	K = 1
	
	print("Building tree for exact K-NN")
	start = time.time()
	# Build the tree for exact K-NN here so we only have to do it ones
	nbrs = NearestNeighbors(n_neighbors = K, algorithm = 'ball_tree').fit(P)
	print("Done building the tree, it took: " + str(time.time() - start) + " seconds")
	
	indices = range(1, len(T) + 1)
	for l in indices:
		start = time.time()
		success = 0.0
		#for test in range(0, tries):
		for queryPoint in queryPoints:
			approx = getApproxNN(P, T[:l], np.array(queryPoint), K)
			exact = getExactNNB(P, np.array(queryPoint), K, nbrs)
		
			for idx, val in enumerate(approx):
				if val == exact[idx]:
					success += 1.0
	
			
		success /= numQueryPoints * K
		error.append(1 - success)
		print("Time taken when using " + str(l) + " hash tables: " + str(time.time() - start) + " seconds")

		
		
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title("Alpha=., n=" + str(len(P)) + ", d=" + str(P.shape[-1]) + ", k=" + str(k))
	ax.set_xlabel("Number of indices")
	ax.set_ylabel("Error")
	ax.plot(indices, error, 'r-')
	ax.plot(indices, error, 'bs')
	ax.axis([0, indices[-1] + 1, 0, max(error) + 0.1])
	plt.show()
		
def figure9():
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



def ExperimentWithDataset2(dataset2):
  import pandas as pd
  datasetFullPath = projectPath + "/data/" + dataset2
  k = 65
  # k = 100
  # k = 700

  B = 25000
  numQueryPoints = 1000
  maxNumberOfHashTables = 10
  numDataPoints = sys.maxint
  # datasetFactor = 1
  # datasetFactor = 10000
  datasetFactor = 100000


  df = pd.read_csv(datasetFullPath)
  df = df.ix[:, 2:62]
  P2 = df.as_matrix() * datasetFactor
  # P2 = df.as_matrix() * 1000000
  # P2 = df.as_matrix()

  print(P2.__class__)
  print(P2.shape)

  # print(P2[0:5, 0:5])
  # print(P2.head())
  lsh2 = LSH(k, B)
  queryPoints2 = [None] * numQueryPoints
  randNums2 = np.random.choice(len(P2), numQueryPoints, replace=False)
  for idx, randNum in enumerate(randNums2):
    queryPoints2[idx] = P2[randNum]
  P2 = np.delete(P2, randNums2, 0)[:numDataPoints]
  print("Size of data set: " + str(len(P2)))
  print("Size of test set: " + str(len(queryPoints2)))
  error2 = []
  print("Start preprocessing of the data")
  start2 = time.time()
  T2 = lsh2.preprocessing(P2, maxNumberOfHashTables)
  print("Preprocessing done, created " + str(maxNumberOfHashTables) + " hash tables in " + str(
    time.time() - start2) + " seconds")
  # output()
  figure4(P2, k, B, T2, queryPoints2, error2)

def ExperimentWithDataset1():
  # global P, lsh, queryPoints, error, T
  # Open and read database into P
  P = np.delete(np.genfromtxt(projectPath + "/data/" + dataSet, delimiter=" ") * 100000, 0, 1).astype(int)
  print(P.__class__)
  print(P.shape)
  lsh = LSH(k, B)
  queryPoints = [None] * numQueryPoints
  randNums = np.random.choice(len(P), numQueryPoints, replace=False)
  for idx, randNum in enumerate(randNums):
    queryPoints[idx] = P[randNum]
  P = np.delete(P, randNums, 0)[:numDataPoints]
  print("Size of data set: " + str(len(P)))
  print("Size of test set: " + str(len(queryPoints)))
  error = []
  print("Start preprocessing of the data")
  start = time.time()
  T = lsh.preprocessing(P, maxNumberOfHashTables)
  print("Preprocessing done, created " + str(maxNumberOfHashTables) + " hash tables in " + str(
    time.time() - start) + " seconds")
  # output()
  figure4(P, k, B, T, queryPoints, error)


# ExperimentWithDataset1()
ExperimentWithDataset2(dataSet2)

