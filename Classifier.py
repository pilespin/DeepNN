#!/usr/bin/python

from Math import *

import csv
import sys
import numpy as np
from random import randint

class Classifier(Math):

	mt			= None
	m 			= 0
	lr 			= 0
	nbInput		= 0
	# nbOutput	= 0
	weight		= []
	loss 		= 0

	def __init__(self, nbInput):
		np.set_printoptions(precision=4)
		self.nbInput = nbInput
		# self.nbOutput = nbOutput

		self.weight = np.ones(self.nbInput, dtype=float)
		self.printInfo()
		self.mt = Math()

	def printInfo(self):
		print("lr: " + str(self.lr))
		print("nbInput: " + str(self.nbInput))
		# print("nbOutput: " + str(self.nbOutput))
		print("weight: " + str(self.weight))
		print("-----------")

	def updateLr(self, j, loss):
		if loss > 0:
			self.weight[j] -= self.lr
		else:
			self.weight[j] += self.lr

	def sigma(self, X, Y, j):

		print (X)
		i = 0
		sigma = 0.0
		while i < self.m:
			print (X[i])
			Htheta = np.sum(self.predict(X[i]))
			sigma += (Htheta - 1) * X[i][j]
			i+=1
		return sigma

	def train(self, X, Y, oposite=False):
		self.m = len(X)
		print (self.m)

		self.loss = 0

		th = []
		for i,data in enumerate(self.weight):
			weightLoss = 0
			# for x in X:
				# print(x)
			sigma = self.sigma(X, Y, i)
			loss = (sigma / self.m)
			weightLoss += loss

			if oposite == True:
				self.updateLr(i, -weightLoss *0.1)
			else:
				self.updateLr(i, weightLoss)
				self.loss += weightLoss
		
		# sys.stdout.write('.')
		# sys.stdout.flush()

	def predict(self, X):
		# m = Math()
		# return m.sigmoid_core(m.sigmoid(X*self.weight).sum())
		tmp = (X*self.weight).sum()
		return self.mt.sigmoid_core(tmp.sum())
		# return tmp.sum()

	################################## GET ##################################

	def getLoss(self):
		return abs(self.loss)

	def getWeight(self):
		return self.weight

	def setLr(self, lr):
		self.lr = lr
