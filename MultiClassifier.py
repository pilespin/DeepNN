#!/usr/bin/python

from Math import *
from Classifier import *

from threading import Thread

class MultiClassifier(Math):

	allClassifier = []
	nbInput = 0
	nbOutput = 0
	nbClassifier = 0
	m = None

	def __init__(self, nbInput, nbOutput):
		self.nbOutput = nbOutput
		self.nbInput = nbInput
		self.m = Math()

	def printInfo(self):
		for i in self.allClassifier:
			i.printInfo()

	def addClassifier(self, number):
		cl = Classifier(self.nbInput, number)
		self.allClassifier.append(cl)
		self.nbClassifier = len(self.allClassifier)

	def predictAll(self, X):
		out = []
		for i,d in enumerate(self.allClassifier):
			out.append(d.predict(X))
		return np.array(out)

	def train(self, X, Y):

		### Without thread
		# allLoss = []
		# for i,d in enumerate(self.allClassifier):
		# 	# print("TRAIN" + str(i))
		# 	loss = d.train(X, Y, i)
			# allLoss.append(loss)

		##########################################################

		thread_list = []
		allLoss = []
		for i,d in enumerate(self.allClassifier):

			t = Thread(target=d.train, args=(X, Y, i))
			t.start()
			thread_list.append(t)

		for thread in thread_list:
			thread.join()

		for i,d in enumerate(self.allClassifier):
			allLoss.append(d.getLoss())

		return np.array(allLoss)

	def getNbClassifier(self):
		return self.nbClassifier

	def getMax(self, X):
		pr = self.predictAll(X)
		# print("PREDICT ALL: " + str(pr))
		return self.m.argMax(pr)

	def setLr(self, lr):
		for i,d in enumerate(self.allClassifier):
			d.setLr(lr)

	def saveWeight(self):
		with open('weight', 'w') as file:
			for i,d in enumerate(self.allClassifier):
				file.write(str(d.getWeight()) + "\n")

	################################## GET ##################################
