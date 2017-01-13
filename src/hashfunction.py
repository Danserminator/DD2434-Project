import numpy as np
import bisect

class Hash_Function:
	def __init__(self, k, d, C, B):
		self.k = k	# Number of elements from {1, ..., d'}
		self.d = d
		self.C = C
		self.B = B	# Bucket size
		self.dp = C * d
		
		# The bits that this hash function looks at
		self.bitPosition = np.random.choice(self.dp, k, replace=False) #np.random.randint(0, self.dp, k)
		
		'''
		self.stuff = [None] * d
		for i in range(0, k):
			index = self.bitPosition[i] / C
			if self.stuff[index] == None:
				self.stuff[index] = [self.bitPosition[i] % C]
			else:
				self.stuff[index].append(self.bitPosition[i] % C)
		
		for i in range(0, d):
			if self.stuff[i] != None:
				# Sort biggest first
				self.stuff[i].sort()
		
		'''
		self.indices = [None] * k
		self.values = [None] * k
		self.twoExp = [None] * k
		for i in range(0, k):
			# This will tell us which of the d coordinates to look at
			self.indices[i] = self.bitPosition[i] / C
			# This will tell the minimum size of the number to be 1
			self.values[i] = self.bitPosition[i] % C
			# Calculate this now instead...
			self.twoExp[i] = 2 ** i
		
		
		# The buckets where the points are stored
		# Hash code : point
		self.bucket = dict()
		
	# Add a new point into a bucket
	def add(self, point, index):		
		# Get the hash code for this point
		hashCode = self.getHashCode(point)
		
		if hashCode in self.bucket:
			# TODO: Check the maximum bucket size
			if len(self.bucket[hashCode]) < self.B:
				self.bucket[hashCode].append(index)
		else:
			self.bucket[hashCode] = [index]
			
			
	# Generate hash code
	def getHashCode(self, data):
		
		hashCode = 0	# TODO: Is this better than a string?!
		for i in range(0, self.k):			
			if data[self.indices[i]] >= self.values[i]:
				hashCode += self.twoExp[i]
		
		return hashCode
		'''
		
		hashCode = [None] * len(data)
		
		for i in range(0, len(data)):
			if self.stuff[i] == None:
				continue
				
			j = bisect.bisect(self.stuff[i], data[i])
					
			hashCode[i] = j
		
				
		return tuple(hashCode)
		'''
		
	# Get all the points in the bucket with the same hash code as this point
	def get(self, point):
		hashCode = self.getHashCode(point)
		
		if hashCode in self.bucket:
			return self.bucket[hashCode]
		else:
			return []
