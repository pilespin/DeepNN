#!/usr/bin/python3

from Math import *

import csv
import sys
import numpy as np
import random

# np.set_printoptions(precision=4)
# np.set_printoptions(suppress=True)

class Classifier(Math):

	mt			= None
	m 			= 0
	lr 			= 0
	nbInput		= 0
	number		= 0
	weight		= []
	loss 		= 0
	nameOutput	= ""

	def __init__(self, nbInput, number, nameOutput):
		self.nbInput = nbInput
		self.number = number
		self.nameOutput = nameOutput

		# self.weight = np.ones(self.nbInput, dtype=float)
		self.weight = np.random.uniform(low=0.999, high=1, size=(self.nbInput,))
		self.mt = Math()

	def initWeight(self, weight):
		if len(self.weight) != len(weight):
			print("Error different size when try to init weight of size " + str(len(weight)) + " instead of " + str(len(self.weight)))
			exit(1)

		self.weight = weight

	def printInfo(self):
		print("classifier " + str(self.number) + ":")
		print("lr: " + str(self.lr))
		print("nbInput: " + str(self.nbInput))
		print("weight: " + str(self.weight))
		print("-----------")

	def updateLr(self, i, loss):
		# print "LOSS: " + str(loss)
		self.weight[i] -= self.lr * loss

		# if loss > 0:
		# 	self.weight[i] -= self.lr
		# else:
		# 	self.weight[i] += self.lr

	def sigma(self, X, Y, classifierNb, thNb):
		classifierNb += 1
		m = 0
		sigma = 0.0
		for i in range(self.m):
			Htheta = self.predict(X[i])

			if classifierNb == Y[i]:
				m += 1
				# sigma += (Htheta - 1) * X[i][thNb]
				sigma += np.log(Htheta) * X[i][thNb]
			# else:
				# m += 1
				# tmp = (Htheta) * X[i][thNb]
				# tmp = np.log(1-Htheta) * X[i][thNb]
				# print "NOT EQUAL: " + str(tmp)

		return sigma / m

	def train(self, X, Y, classifierNb):
		self.m = len(X)
		allLoss = []
		for i,data in enumerate(self.weight):
			sigma = self.sigma(X, Y, classifierNb, i)
			loss = sigma
			allLoss.append(loss)

		for i,loss in enumerate(allLoss):
			self.updateLr(i, loss)

		self.loss = np.array(allLoss).sum()
		return (self.loss)
		
		# sys.stdout.write('.')
		# sys.stdout.flush()

	def predict(self, X):
		tmp = X * self.weight
		return self.mt.sigmoid_core(tmp.sum())

	################################## GET ##################################

	def getOutputName(self):
		return self.nameOutput

	def getLoss(self):
		return abs(self.loss)

	def getWeight(self):
		return self.weight

	def setLr(self, lr):
		self.lr = lr
