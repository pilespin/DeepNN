#!/usr/bin/python

from Math import *

import csv
import sys
import numpy as np
from random import randint

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

	def __init__(self, nbInput, number):
		self.nbInput = nbInput
		self.number = number

		self.weight = np.ones(self.nbInput, dtype=float)
		self.mt = Math()

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
		# print ("INC: " + str(classifierNb))
		m = 0
		sigma = 0.0
		for i in range(self.m):
			Htheta = self.predict(X[i])

			# print ("OUT: " + str(Y[i]) + "EXT: " + str(X[i][thNb]))
			# print "-------"
			# print ("CLASSIFIER: " + str(classifierNb) + " OUTPUT: " + str(Y[i]))

			if classifierNb == Y[i]:
				Htheta = self.predict(X[i])
				m += 1
				tmp = np.log(Htheta) * X[i][thNb]
				sigma += tmp
				# print "    EQUAL: " + str(tmp)
			# else:
			# 	Htheta = self.predict(-X[i])
			# 	# y = 0
			# 	tmp = np.log(Htheta)  * -X[i][thNb]
			# 	sigma += tmp
				# print "NOT EQUAL: " + str(tmp)

				# sigma += (1 - Htheta) 
				# sigma += (Htheta - y) * X[i][thNb]
				# sigma += (1-Htheta) * X[i][thNb]
				# print ("FALSE")

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

	def getLoss(self):
		return abs(self.loss)

	def getWeight(self):
		return self.weight

	def setLr(self, lr):
		self.lr = lr
