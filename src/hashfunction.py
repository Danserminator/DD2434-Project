import numpy as np

class Hash_Function:
	def __init__(self, k, d, C, B):
		self.k = k	# Number of elements from {1, ..., d'}
		self.d = d
		self.C = C
		self.B = B	# Bucket size
		self.dp = C * d
		
		# The bits that this hash function looks at
		self.bitPosition = np.random.choice(self.dp, k, replace=False) #np.random.randint(0, self.dp, k)
		
		# The buckets where the points are stored
		# Hash code : point
		self.bucket = dict()
		
	# Add a new point into a bucket
	def add(self, point, index):		
		# Get the hash code for this point
		hashCode = self.getHashCode(point)
		
		if hashCode in self.bucket:
			# TODO: Check the maximum bucket size
			self.bucket[hashCode].append(index)
		else:
			self.bucket[hashCode] = [index]
			
			
	# Generate hash code
	def getHashCode(self, data):
		hashCode = 0	# TODO: Is this better than a string?!
		for i in range(0, self.k):
			# This will tell us which of the d coordinates to look at
			index = self.bitPosition[i] / self.C
			# This will tell the minimum size of the number to be 1
			m = self.bitPosition[i] % self.C
			
			if data[index] >= m:
				hashCode += 10 ** i
				
		return hashCode
		
		
	# Get all the points in the bucket with the same hash code as this point
	def get(self, point):
		hashCode = self.getHashCode(point)
		
		if hashCode in self.bucket:
			return self.bucket[hashCode]
		else:
			return []
