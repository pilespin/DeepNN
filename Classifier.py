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
	# nbOutput	= 0
	weight		= []
	loss 		= 0

	def __init__(self, nbInput, number):
		# np.set_printoptions(precision=4)
		self.nbInput = nbInput
		self.number = number
		# self.nbOutput = nbOutput

		self.weight = np.ones(self.nbInput, dtype=float)
		self.mt = Math()

	def printInfo(self):
		print("classifier " + str(self.number) + ":")
		print("lr: " + str(self.lr))
		print("nbInput: " + str(self.nbInput))
		# print("nbOutput: " + str(self.nbOutput))
		print("weight: " + str(self.weight))
		print("-----------")

	def updateLr(self, i, loss):
		# print "LOSS: " + str(loss)
		self.weight[i] -= loss

		# if loss > 0:
		# 	self.weight[i] -= self.lr
		# else:
		# 	self.weight[i] += self.lr

	def sigma(self, X, Y, classifierNb, thNb):
		classifierNb += 1
		# print ("INC: " + str(classifierNb))
		# i = 0
		sigma = 0.0
		for i in range(self.m):
			# print "------"
			# print X[i]
			# print self.predict(X[i])

			Htheta = self.predict(X[i])
			
			# print (Htheta)
			# print thNb
			# print(classifierNb)
			# print ("OUT: " + str(Y[i]) + "EXT: " + str(X[i][thNb]))
			# print "-------"
			# print ("CLASSIFIER: " + str(classifierNb) + " OUTPUT: " + str(Y[i]))

			if classifierNb == Y[i]:
				# y = 1
				sigma += (Htheta - 1)
			else:
				# y = 0
				sigma += (1 - Htheta)

				# sigma += (1 - Htheta) 
				# sigma += (Htheta - y) * X[i][thNb]
				# sigma += (1-Htheta) * X[i][thNb]
				# print ("FALSE")

			# sigma += (Htheta - y) * X[i][thNb]
			# i+=1
		# print "---------"
		# print "SIGMA: " + str(sigma)
		# print "---------"
		return sigma

	def train(self, X, Y, classifierNb):
		self.m = len(X)
		# print (self.m)
		allLoss = []
		for i,data in enumerate(self.weight):
			# print "WEIGHT: " + str(i)
			sigma = self.sigma(X, Y, classifierNb, i)
			loss = sigma / self.m
			allLoss.append(loss)
		# print(np.array(allLoss))
		# print allLoss

		# self.loss = 0
		for i,loss in enumerate(allLoss):
		# 	if oposite == True:
				# self.updateLr(i, -loss * 0.1)
		# 	else:
			self.updateLr(i, loss)
		# 		self.loss += loss
		# exit(0)
		return (np.array(allLoss).sum())
		
		# sys.stdout.write('.')
		# sys.stdout.flush()

	def predict(self, X):
		# m = Math()
		# return m.sigmoid_core(m.sigmoid(X*self.weight).sum())
		# tmp = (X*self.weight).sum()
		# print X
		# print self.weight
		# print "X: " + str(X)
		# print "W: " + str(self.weight)
		tmp = X * self.weight
		# print tmp.sum()
		return self.mt.sigmoid_core(tmp.sum())
		# return tmp.sum()

	################################## GET ##################################

	def getLoss(self):
		return abs(self.loss)

	def getWeight(self):
		return self.weight

	def setLr(self, lr):
		self.lr = lr
