from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
from lsh import *
import time

dataSet = "1000histograms.asc"

def getDistanceL2(p1, p2):
	return np.sqrt(np.sum(np.power(np.abs(p1-p2),2)))

def getDistanceL1(p1, p2):
	return np.sum(np.abs(p1-p2))

def getApproxNN(T, point, K):
	S = set()
	for t in T:
		S = S.union(t.get(point))
	
	return getExactNNB(np.array(list(S)),point,K)


def getExactNNB(P,point,K):
	K = np.minimum(K,len(P))
	if K > 0:
		d = list(map((lambda p: (getDistanceL2(p,point),tuple(p.tolist())) ), P))
		dd = sorted(d,key=lambda dist: dist[0])
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
	lsh = LSH(k, B)
	
	numQueryPoints = 90
	queryPoints = [None] * numQueryPoints
	
	for i in range(0, numQueryPoints):
		randNum = np.random.randint(0, len(P))
		queryPoints[i] = P[randNum]
		P = np.delete(P, randNum, 0)
	
	#q = 3
	#queryPoint = P[q]
	#P = np.delete(P, q, 0)
	
	K = 2
	
	#tries = 1
	
	indices = range(1, 11)
	error = []
	for l in indices:
		print(l)
		start = time.time()
		T = lsh.preprocessing(P, l)
		success = 0.0
		#for test in range(0, tries):
		for queryPoint in queryPoints:
			approx = getApproxNN(T, np.array(queryPoint), K)
			exact = getExactNNB(P, np.array(queryPoint), K)
			
			for idx, val in enumerate(approx):
				if val == exact[idx]:
					success += 1.0
		
				
		success /= numQueryPoints * K
		error.append(1 - success)
		print(time.time() - start)

		
		
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title("Alpha=., n=" + str(len(P)) + ", d=" + str(P.shape[-1]) + ", k=" + str(k))
	ax.set_xlabel("Number of indices")
	ax.set_ylabel("Error")
	ax.plot(indices, error, 'r-')
	ax.plot(indices, error, 'bs')
	ax.axis([0, indices[-1] + 1, 0, max(error) + 0.1])
	plt.show()
		
		
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
	
		
# The data "has to be" stored this way 
#P = np.array([[1], [34], [99], [2], [5], [2], [6], [8], [10], [100]])
P = []
 
# Open and read database into P
P = np.delete(np.genfromtxt("../data/" + dataSet,delimiter=" ") * 1000,0,1).astype(int)
#P = np.array([[1, 2], [35, 20], [99, 1], [2, 55], [5, 5], [2, 88]])
#P = np.array([[1,2,3], [4,5,6]])


k = 50
B = 5
'''
lsh = LSH(k, B)

l = 1	# Number of indices

T = lsh.preprocessing(P, l)
'''
#output()

figure4(P, k, B)
