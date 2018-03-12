#!/usr/bin/python3

from Math import *
from Classifier import *

import csv
from threading import Thread

class MultiClassifier(Math):

	def __init__(self, nbInput, allOutput):

		self.allClassifier = []
		self.nbClassifier = 0

		self.nbOutput = len(allOutput)
		self.nbInput = nbInput
		self.m = Math()
		for i,d in enumerate(allOutput):
			self.addClassifier(i, d)

	def initWeight(self, allWeight):
		if len(allWeight) != self.nbClassifier:
			print("Error different size when try to init weight")
			exit(1)

		for i,d in enumerate(self.allClassifier):
			# if allWeight[i] == len(d.getWeight)
			d.initWeight(allWeight[i])


	def printInfo(self):
		for i in self.allClassifier:
			i.printInfo()

	def addClassifier(self, number, nameOutput):
		cl = Classifier(self.nbInput, number, nameOutput)
		self.allClassifier.append(cl)
		self.nbClassifier = len(self.allClassifier)

	def predictAll(self, X):
		out = []
		for i,d in enumerate(self.allClassifier):
			out.append(d.predict(X))
		return np.array(out)

	def weightRegularization(self, threshold):
		for i in self.allClassifier:
			if np.min(i.getWeight()) < threshold:
				break

			for i in self.allClassifier:
				a = [num-1 for num in i.getWeight()]
				i.initWeight([a])

	def train(self, X, Y):

		self.weightRegularization(3)

		### Without thread
		allLoss = []
		for i,d in enumerate(self.allClassifier):
			# print("TRAIN" + str(i))
			loss = d.train(X, Y, i)
			allLoss.append(loss)

		##########################################################

		# thread_list = []
		# allLoss = []
		# for i,d in enumerate(self.allClassifier):

		# 	t = Thread(target=d.train, args=(X, Y, i))
		# 	t.start()
		# 	thread_list.append(t)

		# for thread in thread_list:
		# 	thread.join()

		# for i,d in enumerate(self.allClassifier):
		# 	allLoss.append(d.getLoss())

		return np.array(allLoss)

	def getNbClassifier(self):
		return self.nbClassifier

	def getMax(self, X):
		pr = self.predictAll(X)
		# print("PREDICT ALL: " + str(pr))
		return self.m.argMax(pr)

	def predict(self, X):
		pr = self.predictAll(X)
		# print("PREDICT ALL: " + str(pr))
		ret = self.m.argMax(pr)
		name = self.allClassifier[ret].getOutputName()
		return name

	def setLr(self, lr):
		for i,d in enumerate(self.allClassifier):
			d.setLr(lr)

	def saveWeight(self):
		All = []
		for i,d in enumerate(self.allClassifier):
			name = d.getOutputName()
			weight = d.getWeight()

			tmp = []
			tmp.append(name)
			for i in weight:
				tmp.append(i)
			All.append(tmp)

		with open('weight.csv', 'w') as file:
			csvWriter = csv.writer(file, delimiter=',')
			csvWriter.writerows(All)
