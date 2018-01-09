#!/usr/bin/python

import csv
import math
import numpy as np

class Math(object):

	def __init__(self):
		pass

	def sigmoid_core(self, number):
		if number >= 0:
			nb = (math.e * number) / ((math.e * number) + 1)
			return nb
		elif number < 0:
			nb = 1 / (1 + (math.e - number))
			return nb
		exit(1)

	def sigmoid(self, X):
		Y = np.array([])
		for i in X:
			Y = np.append(Y, self.sigmoid_core(i))
		return Y

	def mean(self, X):
		s = 0
		i = 0
		for x in X:
			i+=1
			s += (x - s) / i
		return s

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
