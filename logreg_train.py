#!/usr/bin/python

from Dataset import *
from Classifier import *
from MultiClassifier import *
from Math import *

import numpy as np
import csv
import sys	

from sklearn.metrics import accuracy_score

# import matplotlib.pyplot as plt

# np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

def checkArg(argv):
	if len(sys.argv) <= 1:
		print("Missing file")
		exit(1)

	file = sys.argv[1]

	try:
		open(file, 'r')
	except IOError:
		print("Can't read: " + file)
		exit(1)
	return (file)

def csvToArray(file):
	file = open(file, "r")
	arr = csv.reader(file, delimiter=',')

	X = []
	Y = []

	i = 0
	for line in arr:
		i+=1
		if i!=1:
			X.append(line[0])
			Y.append(line[1])
	
	if len(X) != len(Y):
		print("Error")
		exit(1)
	
	return (X, Y)

#############################################################

def getHouseByIndex(d, index):
	house = d.getDataset()[index][1]
	return house

def getIndex(X, querie):
	i = 0
	for x in X:
		i+=1
		if x == querie:
			return int(i)
	return -1

def getInputInDataset(d, index, inFloat=False):
	global nbInput
	global start
	end = start + nbInput - 1

	X = []
	if inFloat == True:
		for i in range(start, end+1):
			# print i
			# exit(0)
			# while start <= end:
			tmp = d.getDataset()[index][i]
			# print tmp
			if len(tmp) > 0:
				X.append(float(tmp))
			else:
				X.append(float(0))
			# start += 1
		# print X
		# exit(0)
	else:
		for i in range(start, end+1):
			X.append(d.getDataset()[index][i])

	return np.array(X)

def generateDataset(d, index=-1):

	X = []
	Y = []

	houseArray = d.getFeature(1, uniq=True)

	for i in range(d.getLength()):
		x = getInputInDataset(d, i, inFloat=True)
		y = getIndex(houseArray, getHouseByIndex(d, i))

		if index == -1 or (y == index):
			X.append(x)
			Y.append(y)

	X = np.array(X)
	Y = np.array(Y)
	if len(X) != len(Y):
		print("Error when generate dataset")
		exit(1)
	return X, Y

def generatePrediction(allclassifier, X, Y):
	y_pred = []
	y_true = []
	m = Math()

	for i,data in enumerate(X):
		output = allclassifier.getMax(data) + 1
		# print "PREDICT: " + str(output)
		# print Y[i][0]

		y_pred.append(output)
		y_true.append(Y[i])

	if len(y_true) != len(y_pred):
		print("Error when generate prediction")
		exit(1)

	return np.array(y_true), np.array(y_pred)


##############################
############ MAIN ############
##############################

nbInput = 3
nbOutput = 4
start = 6

def main():

	file = checkArg(sys.argv)

	d = Dataset()
	d.loadFile(file)

	allclassifier = MultiClassifier(nbInput, nbOutput)

	X, Y = generateDataset(d)

	print X
	print "------------"
	print Y
	print "------------"

	for i in range(nbOutput):
		allclassifier.addClassifier(i)

	lr = 1.0
	oldLoss = 0
	allclassifier.setLr(lr)

	allclassifier.printInfo()

	for j in range(20000):
		loss = allclassifier.train(X, Y)

		allLoss = loss.sum()

		if abs(oldLoss) > abs(allLoss) and lr > 0.000000001:
			lr /= 10
			print("DECREASE TO " + str(lr))
			allclassifier.setLr(lr)
		oldLoss = allLoss

		allclassifier.saveWeight()
		
		y_true, y_pred = generatePrediction(allclassifier, X, Y)

		# print(y_true)
		# print(y_pred)

		acc = accuracy_score(y_true, y_pred) * 100
		print("epoch: {0:<15.5g} Loss1: {1:<15.5g} Loss2: {2:<15.5g} Loss3: {3:<15.5g} Loss4: {4:<15.5g} LOSS: {5:<15.5g} Accuracy: {6:<g}%" \
		.format(j, loss[0], loss[1], loss[2], loss[3], allLoss, acc))


main()
