#!/usr/bin/python3

import numpy as np
import csv
import sys

sys.path.append('Class')
from Dataset import *
from Classifier import *
from MultiClassifier import *
from Math import *
from IOHelper import *

from sklearn.metrics import accuracy_score

# np.set_printoptions(precision=4)
np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)

#############################################################

def getHouseByIndex(d, index):
	house = d.getDataset()[index][1]
	return house

def getIndex(X, querie):
	for i,x in enumerate(X):
		if x == querie:
			return int(i+1)
	return -1

def getInputInDataset(d, index, featuresId, inFloat=False):
	X = []
	if inFloat == True:
		for i in featuresId:
			tmp = d.getDataset(index, i)
			if len(tmp) > 0:
				X.append(float(tmp))
			else:
				return None
	else:
		for i in featuresId:
			X.append(d.getDataset(index, i))

	return np.array(X)

def generateDataset(d, featuresId, index=-1):

	X = []
	Y = []

	houseArray = d.getFeature(1, uniq=True)

	for i in range(d.getLength()):
		x = getInputInDataset(d, i, featuresId, inFloat=True)
		y = getIndex(houseArray, getHouseByIndex(d, i))

		if x is not None: 
			if index == -1 or (y == index):
				if len(x) > 0 and y >= 0:
					X.append(x)
					Y.append(y)

	X = np.array(X)
	Y = np.array(Y)
	if len(X) != len(Y):
		print("Error when generate dataset")
		exit(1)
	if len(X) <= 0:
		print("Error Empty dataset")
		exit(1)
	return X, Y

def generatePrediction(allclassifier, X, Y):
	y_pred = []
	y_true = []

	for i,data in enumerate(X):
		output = allclassifier.getMax(data) + 1

		y_pred.append(output)
		y_true.append(Y[i])

	if len(y_true) != len(y_pred):
		print("Error when generate prediction")
		exit(1)

	return np.array(y_true), np.array(y_pred)

##############################
############ MAIN ############
##############################

def main():

	nbInput = 0
	nbOutput = 0
	epoch = 30

	file = IOHelper().checkArg(sys.argv)
	if (len(file) < 1):
		print "Missing file"
		exit(1)

	d = Dataset()
	d.loadFile(file[0])

	featuresId = range(7, 19)
	# nbInput = len(featuresId)
	X, Y = generateDataset(d, featuresId)

	X, nbInput = d.featureExpand(d, X)
	X = d.featureRescale(d, X)

	houseArray = d.getFeature(1, uniq=True)
	nbOutput = len(houseArray)

	allclassifier = MultiClassifier(nbInput, houseArray)

	lr = 1000.0
	oldLoss = 9e+9
	allclassifier.setLr(lr)

	# allclassifier.printInfo()

	for j in range(epoch):
		loss = allclassifier.train(X, Y)

		allLoss = loss.sum()

		if abs(allLoss) > abs(oldLoss) and lr > 0.000000001:
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
