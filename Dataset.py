#!/usr/bin/python

import csv
import math

class Dataset(object):

	_X = []
	_Nm = []

	def __init__(self):
		pass

	################################## GET ##################################

	def getDataset(self):
		return (_X)

	def getName(self, index):
		for x in self._Nm:
			if len(x[index]) > 0:
				return x[index]
		return None

	def getFeature(self, index):
		X = []
		for x in self._X:
			if len(x[index]) > 0:
				X.append(float(x[index]))

		return X

	################################## INIT ##################################

	def loadFile(self, file):
		file = open(file, "r")
		arr = csv.reader(file, delimiter=',')

		i = 0
		for line in arr:
			if i == 0:
				i+=1
				self._Nm.append(line)
				continue
			self._X.append(line)
		return (self._X, self._Nm)

	################################## CALC ##################################

	def count(self, index):
		i = 0
		for x in self._X:
			if len(x[index]) > 0:
				i+=1
		return i

	def mean(self, index):
		s = 0
		i = 0
		for x in self._X:
			if len(x[index]) > 0:
				i+=1
				add = float(x[index])
				s += (add - s) / i
		return s

	def standardDeviation(self, index, mean):
		s = 0
		i = 0
		for x in self._X:
			if len(x[index]) > 0:
				i+=1
				add = math.pow((float(x[index]) - mean), 2)
				s += (add - s) / i
		return math.sqrt(s)

	def min(self, index):
		m = None
		for x in self._X:
			if len(x[index]) > 0:
				new = float(x[index])
				if (m == None):
					m = new
				elif new < m:
					m = new
		return m

	def max(self,index):
		m = None
		for x in self._X:
			if len(x[index]) > 0:
				new = float(x[index])
				if (m == None):
					m = new
				elif new > m:
					m = new
		return m

	def medianArray(self, X):
		m = len(X)
		if m <= 0:
			print "Error when getting quartile array size of " + str(m)	
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

	def quartile(self, index):
		X = []
		for x in self._X:
			if len(x[index]) > 0:
				X.append(float(x[index]))

		X.sort()
		A, med, B = self.medianArray(X)
		n1, one, n2 = self.medianArray(A)
		n1, two, n2 = self.medianArray(B)

		return one, float(med), two

	################################## PRINT ##################################

	def printFeatureHeader(self): 
		print "                                         Count           Std             Mean            Min             25%             50%             75%             Max"

	def printAllFeature(self):
		for i in range(len(self._X[0])):
			self.printFeature(i)

	def printFeature(self, index):
		if self._isFloat(self._X[0][index]) == False or self.getName(index) == "Index":
			return;
	
		nom = self.getName(index)
		nb = self.count(index)
		moy = self.mean(index)
		std = self.standardDeviation(index, moy)
		min1 = self.min(index)
		q25, q50, q75 = self.quartile(index)
		max1 = self.max(index)
	 
		print("{0:<40s} {1:<15.5g} {2:<15.5g} {3:<15.5g} {4:<15.5g} {5:<15.5g} {6:<15.5g} {7:<15.5g} {8:<15.5g}" \
			.format(nom, nb, std, moy, min1, q25, q50, q75, max1))

	################################## ELSE ##################################

	def _isFloat(self, string):
		try:
			x = float(string)
			return(True)
		except ValueError:
			return(False)
		return(False)
