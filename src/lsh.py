import numpy as np
from hashfunction import *

class LSH:
	def __init__(self, k, B):
		self.k = k	# Number of elements from {1, ..., d'}
		self.B = B	# Bucket size
		
	def preprocessing(self, P, l):
		d = self.getDimension(P)
		C = self.getLargest(P)
		dp = C * d
	
		T = [None] * l
	
		for i in range(0, l):
			# Initialize hash table T_i by generating a random hash function gi(*)
			T[i] = Hash_Function(self.k, d, C, self.B)
			
			for j in range(0, len(P)):
				# Store point p_j on bucket g_i(p_j) of hash table T_i
				T[i].add(P[j], j)
		
		return T
		
	# Get the largest element in the data
	def getLargest(self, data):
		return np.ceil(np.amax(data)).astype(int)
		# return np.amax(data)
		
	# Get the dimesion of the data
	def getDimension(self, data):
		if len(data.shape) == 1:
			return 1
		else:
			return data.shape[-1]
