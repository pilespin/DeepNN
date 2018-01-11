#!/usr/bin/python

import numpy as np
import csv
import sys	
# import time
from Dataset import *
from Classifier import *
from MultiClassifier import *
from Math import *
from sklearn.metrics import accuracy_score

# import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

def checkArg(argv):
	if len(sys.argv) <= 1:
		print "Missing file"
		exit(1)

	file = sys.argv[1]

	try:
		open(file, 'r')
	except IOError:
		print "Can't read: " + file
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
		print "Error"
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
	start = 6
	end = 18

	X = []
	if inFloat == True:
		while start <= end:
			tmp = d.getDataset()[index][start]
			if len(tmp) > 0:
				X.append(float(d.getDataset()[index][start]))
			else:
				X.append(float(0))
			start += 1
	else:
		while start <= end:
			Xd.append(d.getDataset()[index][start])
			start += 1
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
			Y.append([y])

	X = np.array(X)
	Y = np.array(Y)
	if len(X) != len(Y):
		print "Error when generate dataset"
		exit(1)
	return X, Y

def generatePrediction(allclassifier, X, Y):
	y_pred = []
	y_true = []
	m = Math()

	for i,x in enumerate(X):
		for j,y in enumerate(x):
			output = allclassifier.getMax(y) + 1
			y_pred.append(output)
			y_true.append(Y[i][0])

	if len(y_true) != len(y_pred):
		print "Error when generate prediction"
		exit(1)

	return y_true, y_pred


##############################
############ MAIN ############
##############################

def main():

	nbInput = 13
	nbOutput = 4

	file = checkArg(sys.argv)

	m = Math()
	d = Dataset()
	d.loadFile(file)

	allclassifier = MultiClassifier(nbInput, nbOutput)

	X = [None]*nbOutput
	Y = [None]*nbOutput

	for i in range(nbOutput):
		X[i], Y[i] = generateDataset(d, index=i+1)
		allclassifier.addClassifier(X[i], Y[i])

	print "train"
	y_true = [None]*nbOutput
	y_pred = [None]*nbOutput

	for j in range(1):
		loss = allclassifier.train()
		mean = allclassifier.predictAll(X[0])

		y_true[i], y_pred[i] = generatePrediction(allclassifier, X, Y)
		acc = accuracy_score(y_true[i], y_pred[i]) * 100
		# acc = str(acc)
		# print "epoch " + str(j) + " Accuracy: " + str(acc) + "%"

		print("epoch: {0:<15.5g} Loss1: {1:<15.5g} Loss2: {2:<15.5g} Loss3: {3:<15.5g} Loss4: {4:<15.5g} Accuracy: {5:<15.5g}" \
		.format(j, loss[0], loss[1], loss[2], loss[3], acc))


main()
