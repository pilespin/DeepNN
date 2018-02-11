#!/usr/bin/python

import csv
import math
import numpy as np

class Math(object):

	def __init__(self):
		pass

	def sigmoid_core(self, number):
		if number >= 0:
			tmp = math.e * number
			nb = tmp / (tmp + 1)
		else:
			nb = 1 / (1 + (math.e - number))
		return nb

	def sigmoid(self, X):
		Y = []
		for i in X:
			Y.append(self.sigmoid_core(i))
		return np.array(Y)

	def mean(self, X):
		s = 0
		i = 0
		for x in X:
			i+=1
			s += (x - s) / i
		return s

	def count(self, X):
		return len(X)
		# i = 0
		# for x in X:
		# 	i+=1
		# return i

	def standardDeviation(self, X, mean):
		s = 0
		i = 0
		for x in X:
			i+=1
			add = (x - mean) **2
			s += (add - s) / i
		return np.sqrt(s)

	def min(self, X):
		m = None
		for new in X:
			if (m == None):
				m = new
			elif new < m:
				m = new
		return m

	def max(self, X):
		m = None
		for new in X:
			if (m == None):
				m = new
			elif new > m:
				m = new
		return m

	def moy2D(self, X):
		allMoy = []

		for i in X:
			allMoy.append(self.mean(i))

		return self.max(allMoy)

	def min2D(self, X):
		allMin = []

		for i in X:
			allMin.append(self.min(i))

		return self.min(allMin)

	def max2D(self, X):
		allMax = []

		for i in X:
			allMax.append(self.max(i))

		return self.max(allMax)

	def std2D(self, X):
		allStd = []
		allMoy = []

		for i in X:
			allMoy.append(self.mean(i))

		moy = self.mean(allMoy)

		for i in X:
			allStd.append(self.standardDeviation(i, moy))

		return self.min(allStd)

	def medianArray(self, X):
		m = len(X)
		if m <= 0:
			print("Error when getting quartile array size of " + str(m))
			exit(1)
		if m == 1:
			return [X[0]], [X[0]], X[0]
		if m % 2 == 0:
			# print "IS PAIR"
			first = m / 2
			second = m / 2
			med = (X[first]-1 + X[second]) / 2.0
			a = X[:first]
			b = X[second:m]
			return a, med, b
		else:
			# print "IS IMPAIR"
			first = ((m+1) / 2) - 1
			second = ((m+1) / 2)
			med = (X[first])
			a = X[:first]
			b = X[second:m]
			return a, med, b

	def quartile(self, X):
		X.sort()
		A, med, B = self.medianArray(X)
		n1, one, n2 = self.medianArray(A)
		n1, two, n2 = self.medianArray(B)

		return one, float(med), two

	def argMax(self, X):
		m = None
		ret = -1
		for idx,new in enumerate(X):
			if (m == None):
				m = new
				ret = idx
			elif new > m:
				m = new
				ret = idx
		return ret
