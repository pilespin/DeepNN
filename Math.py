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
