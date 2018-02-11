#!/usr/bin/python3

from Dataset import *
from Classifier import *
from MultiClassifier import *
from Math import *
from IOHelper import *

import numpy as np
import csv
import sys	

from sklearn.metrics import accuracy_score

# np.set_printoptions(precision=4)
np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)

def csvToArray(file):
	file = open(file, "r")
	arr = csv.reader(file, delimiter=',')

	X = []
	Y = []

	for i,line in enumerate(arr):
		if i != 0:
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
	for i,x in enumerate(X):
		if x == querie:
			return int(i+1)
	return -1

def getInputInDataset(d, index, featuresId, inFloat=False):
	global nbInput

	X = []
	if inFloat == True:
		for i in featuresId:
			tmp = d.getDataset()[index][i]
			if len(tmp) > 0:
				X.append(float(tmp))
			else:
				return None
				# X.append(float(1))
	else:
		for i in featuresId:
			X.append(d.getDataset()[index][i])

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

	for i,data in enumerate(X):
		output = allclassifier.getMax(data) + 1

		y_pred.append(output)
		y_true.append(Y[i])

	if len(y_true) != len(y_pred):
		print("Error when generate prediction")
		exit(1)

	return np.array(y_true), np.array(y_pred)

def rescaleCore(x, min, max):
	ret = 1.0*(((x - (min)) / (max - min)))
	return ret

def meanNormalization(x, moy, min, max):
	ret = 1.0*(((x - (moy)) / (max - min)))+0.5
	return ret

def standardization(x, moy, std):
	ret = 1.0*(((x - (moy)) / (std)))+0.5
	return ret

def featureRescale(d, X):
	min = d.min2D(X)
	max = d.max2D(X)
	# moy = d.moy2D(X)
	# std = d.std2D(X)
	newX = []

	for i,data1 in enumerate(X):
		for j,data2 in enumerate(data1):
			X[i][j] = rescaleCore(data2, min, max)
			# X[i][j] = meanNormalization(data2, moy, min, max)
			# tmp = standardization(data2, moy, std)
			# X[i][j] = rescaleCore(tmp, -30, 30)
	return X

def featureExpand(d, X):
	global nbInput
	newX = []

	for i,data1 in enumerate(X):
		tmp = []
		for j in data1:
			tmp.append(j)

		# for i in range(3):
		tmp.append(1) # intercept

		l = len(X[0])
		for k in range(l):
			# tmp.append(1) # intercept

			# for j in tab:
			for j in range(l):
				pass
				# if j+1 != k:
					# tmp.append(data1[k]*data1[(j+1)%l])

		for i in range(5):
			tmp.append(1)

		nbInput = len(tmp)
		newX.append(tmp) 
	return np.array(newX)


##############################
############ MAIN ############
##############################

nbInput = 0
nbOutput = 4
epoch = 30

def main():

	file = IOHelper().checkArg(sys.argv)

	d = Dataset()
	d.loadFile(file)

	featuresId = range(7, 19)
	# nbInput = len(featuresId)
	X, Y = generateDataset(d, featuresId)

	X = featureExpand(d, X)
	X = featureRescale(d, X)

	allclassifier = MultiClassifier(nbInput, nbOutput)

	for i in range(nbOutput):
		allclassifier.addClassifier(i)

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
