#!/usr/bin/python

import csv
# import math
import numpy as np
from Math import *

class Dataset(Math):

	_X = []
	_Nm = []

	def __init__(self):
		pass

	################################## GET ##################################

	def getDataset(self):
		return (self._X)

	def getName(self, index):
		for x in self._Nm:
			if len(x[index]) > 0:
				return x[index]
		return None

	def getFeature(self, index, column=-1, name='', uniq=False):
		X = []
		if column != -1:
			for x in self._X:
				if len(x[index]) > 0 and x[column] == name:
					X.append(float(x[index]))
		else:
			if self._isFloat(self._X[0][index]) == True:
				for x in self._X:
					if len(x[index]) > 0:
						X.append(float(x[index]))
			else:
				for x in self._X:
					if len(x[index]) > 0:
						X.append(x[index])
		if uniq == True:
			return np.unique(X)
		else:
			return np.array(X)

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


	def countByIndex(self, index):
		# return len(X)
		i = 0
		for x in self._X[index]:
			i+=1
		return i

	def count(self, X):
		return len(X)
		# i = 0
		# for x in X:
		# 	i+=1
		# return i

	def mean(self, X):
		s = 0
		i = 0
		for x in X:
			i+=1
			s += (x - s) / i
		return s

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

	def quartile(self, X):
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
		nb = self.count(self.getFeature(index))
		moy = self.mean(self.getFeature(index))
		std = self.standardDeviation(self.getFeature(index), moy)
		min1 = self.min(self.getFeature(index))
		q25, q50, q75 = self.quartile(self.getFeature(index))
		max1 = self.max(self.getFeature(index))
	 
		print("{0:<40s} {1:<15.5g} {2:<15.5g} {3:<15.5g} {4:<15.5g} {5:<15.5g} {6:<15.5g} {7:<15.5g} {8:<15.5g}" \
			.format(nom, nb, std, moy, min1, q25, q50, q75, max1))

	################################## ELSE ##################################

	def _isFloat(self, string):
		if string != None or len(string) > 0:
			try:
				x = float(string)
				return(True)
			except ValueError:
				return(False)
		return(False)
