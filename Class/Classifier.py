#!/usr/bin/python3

from Math import *

import csv
import sys
import numpy as np
import random

# np.set_printoptions(precision=4)
# np.set_printoptions(suppress=True)

class Classifier(Math):

	def __init__(self, nbInput, number, nameOutput):
		self.mt			= None
		self.m 			= 0
		self.lr 		= 0
		self.nbInput	= 0
		self.number		= 0
		self.weight		= []
		self.nbLayer	= 1
		self.bias 		= 0.00
		self.loss 		= 0
		self.nameOutput	= ""

		self.nbInput 	= nbInput
		self.number 	= number
		self.nameOutput = nameOutput
		self.weight		= []

		# self.weight = np.ones(self.nbInput, dtype=float)
		# self.weight.append(np.ones(self.nbInput, dtype=float))
		for i in range(self.nbLayer):
			self.weight.append(np.ones(self.nbInput, dtype=float))

		self.weight = np.array(self.weight)
		# exit(0)
		# self.weight = np.random.uniform(low=0.9, high=1, size=(self.nbInput,))
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
		self.weight[0][i] -= self.lr * loss

		# if np.min(self.weight[0]) > 1:
		# 	self.weight[0] = [num-1 for num in self.weight[0]]

		# self.weight[0][i] -= self.lr * loss
		# self.weight[1][i] -= self.lr * loss

		# if loss > 0:
		# 	self.weight[0][i] -= self.lr
		# else:
		# 	self.weight[0][i] += self.lr

	def sigma(self, X, Y, classifierNb, thNb):
		classifierNb += 1
		m = 0
		sigma = 0.0
		for i in range(self.m):
			Htheta = self.predict(X[i])

			if classifierNb == Y[i]:
				m += 1
				sigma += (Htheta - 1) * X[i][thNb]
				# sigma += np.log(Htheta) * X[i][thNb]
			# else:
				# m += 1
				# sigma += (Htheta) * X[i][thNb]
				# tmp = np.log(1-Htheta) * X[i][thNb]
				# print "NOT EQUAL: " + str(tmp)

		return sigma / m

	def train(self, X, Y, classifierNb):
		self.m = len(X)
		allLoss = []
		for i,data in enumerate(self.weight[0]):
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

		b = random.uniform(-self.bias, self.bias)
		
		X = (X * self.weight[0]) + b
		# X = self.mt.sigmoid(X)
		# X = (X * self.weight[1]) + b
		# X = self.mt.sigmoid(X)
		return self.mt.sigmoid_core(X.sum())

		# new_X = X
		# for i in range(self.nbLayer):
			# print("-------------------------" + str(i))
			# print(str(X))
			# print(self.weight[i])
			# X = (X * self.weight[i]) + random.uniform(-self.bias, self.bias)
			# if i is not self.nbLayer-1:
			# X = self.mt.sigmoid(X)

			# print("-------------")
			# print(X)
			# print(self.mt.sigmoid(X))
			# print("-------------")
			# exit(0)
			# X = np.array([self.mt.sigmoid_core(X.sum())]*len(X))
			# print("-------------------------")
			# X = X * self.weight[i]
			# print("-------------------------")
			# print(str(X))
			# print(str(self.weight[i]))
			# exit(0)
		# print(self.mt.sigmoid_core(tmp.sum()))
		# exit(0)
		# return X.sum()
		# return self.mt.sigmoid_core(X.sum())

	################################## GET ##################################

	def getOutputName(self):
		return self.nameOutput

	def getLoss(self):
		return abs(self.loss)

	def getWeight(self):
		return self.weight[0]

	def setLr(self, lr):
		self.lr = lr
