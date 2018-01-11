#!/usr/bin/python

from Math import *

import csv
import sys
import numpy as np

class Classifier(Math):

	m 			= 0
	lr 			= 0.00001
	nbInput		= 0
	nbOutput	= 0
	weight		= []
	loss 		= 0

	def __init__(self, nbInput, nbOutput):
		np.set_printoptions(precision=4)
		self.nbInput = nbInput
		self.nbOutput = nbOutput

		self.weight = np.zeros(self.nbInput, dtype=float)
		self.printInfo()

	def printInfo(self):
		print("lr: " + str(self.lr))
		print("nbInput: " + str(self.nbInput))
		print("nbOutput: " + str(self.nbOutput))
		print("weight: " + str(self.weight))
		print("-----------")

	def updateLr(self, j, loss):
		if loss > 0:
			self.weight[j] -= self.lr
		else:
			self.weight[j] += self.lr

	def sigma(self, X, Y, j):

		i = 0
		sigma = 0.0
		while i < self.m:
			Htheta = np.sum(self.predict(X[i]))
			sigma += (Htheta - 1) * X[i][j]
			i+=1
		return sigma

	def train(self, X, Y):
		self.m = len(X)

		self.loss = 0
		for i,data in enumerate(self.weight):
			for x in X:
				sigma = self.sigma(X, Y, i)
				loss = (sigma / self.m)
				self.loss += loss
				self.updateLr(i, loss)
		return self.loss
		# sys.stdout.write('.')
		# sys.stdout.flush()
		# print("")

	def predict(self, X):
		m = Math()
		return m.sigmoid_core(m.sigmoid(X*self.weight).sum())

	################################## GET ##################################

	def getLoss(self):
		return self.loss

	def getWeight(self):
		return self.weight

	def setLr(self, lr):
		self.lr = lr
