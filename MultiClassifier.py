#!/usr/bin/python

from Math import *
from Classifier import *

from threading import Thread

class MultiClassifier(Math):

	allClassifier = []
	nbInput = 0
	nbOutput = 0
	X_train = []
	Y_train = []
	nbClassifier = 0
	m = None

	def __init__(self, nbInput, nbOutput):
		np.set_printoptions(precision=4)
		np.set_printoptions(suppress=True)
		self.nbOutput = nbOutput
		self.nbInput = nbInput
		self.m = Math()

	def addClassifier(self, X, Y):
		cl = Classifier(self.nbInput)
		self.allClassifier.append(cl)
		self.X_train.append(X)
		self.Y_train.append(Y)
		self.nbClassifier = len(self.allClassifier)

	def predictAll(self, X):
		out = []
		for i,d in enumerate(self.allClassifier):
			Y = []
			for x in X:
				Y.append(d.predict(x))
				# print(d.predict(x))
			out.append(self.m.mean(Y))
			# print(Y)		
		return np.array(out)

	def train(self):

		for i,d in enumerate(self.allClassifier):
			d.train(self.X_train[i], self.Y_train[i])

		# print (self.X_train[0])
		# print (self.X_train[1])
		# print (self.X_train[2])
		# print (self.X_train[3])
		# exit(0)

		# thread_list = []
		# allLoss = []
		# for i,d in enumerate(self.allClassifier):

		# 	a = list(range(0, self.getNbClassifier()))
		# 	a.remove(i)
		# 	for j in a:
		# 		d.train(self.X_train[j], self.Y_train[j], oposite=True)
		# 		# d.train(self.X_train[j], self.Y_train[j], oposite=True)

		# 	# print(str("--") + str(i) + str(self.Y_train[i]))
		# 	t = Thread(target=d.train, args=(self.X_train[i], self.Y_train[i]))
		# 	# t = Thread(target=d.train, args=(self.X_train[i], self.Y_train[i]))
		# 	# d.train(self.X_train[i], self.Y_train[i], oposite=True)
		# 	t.start()
		# 	# t.join()
		# 	thread_list.append(t)

		# for thread in thread_list:
		# 	thread.join()
		# for i,d in enumerate(self.allClassifier):
		# 	allLoss.append(d.getLoss())

		# return np.array(allLoss)
		return np.array(0)

	def getNbClassifier(self):
		return self.nbClassifier

	def getMax(self, X):
		pr = self.predictAll(X)
		return self.m.argMax(pr)

	def setLr(self, lr):
		for i,d in enumerate(self.allClassifier):
			d.setLr(lr)

	def saveWeight(self):
		with open('weight', 'w') as file:
			for i,d in enumerate(self.allClassifier):
				file.write(str(d.getWeight()) + "\n")

	################################## GET ##################################
