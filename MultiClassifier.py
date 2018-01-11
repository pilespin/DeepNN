#!/usr/bin/python

from Math import *
from Classifier import *

class MultiClassifier(Math):

	allClassifier = []
	nbInput = 0
	nbOutput = 0
	X_train = []
	Y_train = []
	m = None
	
	def __init__(self, nbInput, nbOutput):
		np.set_printoptions(precision=4)
		self.nbOutput = nbOutput
		self.nbInput = nbInput
		self.m = Math()

	def addClassifier(self, X, Y):
		cl = Classifier(self.nbInput, self.nbOutput)
		self.allClassifier.append(cl)
		self.X_train.append(X)
		self.Y_train.append(Y)

	def predictAll(self, X):
		# m = Math()
		out = []
		for i,d in enumerate(self.allClassifier):
			Y = []
			for x in X:
				Y.append(d.predict(x))
			out.append(self.m.mean(Y))
		return np.array(out)

	def train(self):
		allLoss = []
		tmp = 0
		for i,d in enumerate(self.allClassifier):
			tmp += d.train(self.X_train[i], self.Y_train[i])
			allLoss.append(tmp)
		return np.array(allLoss)

	def getMax(self, X):
		pr = self.predictAll(X)
		return self.m.argMax(pr)


	################################## GET ##################################
