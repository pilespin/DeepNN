#!/usr/bin/python3

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
np.set_printoptions(threshold='nan')
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

def getInputInDataset(d, index, featuresId, inFloat=False):
	global nbInput
	# global start
	# end = start + nbInput - 1

	X = []
	if inFloat == True:
		for i in featuresId:
		# for i in range(start, end+1):
			tmp = d.getDataset()[index][i]
			if len(tmp) > 0:
				X.append(float(tmp))
			else:
				X.append(float(1))
	else:
		for i in featuresId:
		# for i in range(start, end+1):
			X.append(d.getDataset()[index][i])

	return np.array(X)

def generateDataset(d, featuresId, index=-1):

	X = []
	Y = []

	houseArray = d.getFeature(1, uniq=True)

	for i in range(d.getLength()):
		x = getInputInDataset(d, i, featuresId, inFloat=True)
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

		y_pred.append(output)
		y_true.append(Y[i])

	if len(y_true) != len(y_pred):
		print("Error when generate prediction")
		exit(1)

	return np.array(y_true), np.array(y_pred)

def rescaleCore(x, min, max):
	m = Math()
	# x = np.log(x)*10000
	ret = 1.0*(((x - (min)) / (max - min)))
	return ret

def meanNormalization(x, moy, min, max):
	m = Math()
	ret = 1.0*(((x - (moy)) / (max - min)))+0.5
	return ret

def standardization(x, moy, std):
	m = Math()
	# x = np.log(x)
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

		# tab = [2,3,4,5,7,8,11,12]
		# tab = [4,5,7]
		# tab = [1,5,6,8,10,11,12]

		# l = len(tab)
		# for k in tab:

		l = len(X[0])
		for k in range(l):

			# for j in range(tab):
			for j in range(l):
				# pass
				# print str(k) + " * " + str(j+1)
				# if j+1 != k:
					tmp.append(data1[k]*data1[j%l])
					# tmp.append((data1[k]*data1[j%l]*data1[(j+1)%l]))
					# tmp.append(data1[k]*data1[j+2])
				# if j+2 != k:
					# tmp.append(data1[k]*data1[j+2])

		nbInput = len(tmp)
		newX.append(tmp) 
		# exit(0)

	return np.array(newX)

##############################
############ MAIN ############
##############################

nbInput = 0
nbOutput = 4

def main():

	file = checkArg(sys.argv)

	d = Dataset()
	d.loadFile(file)

	featuresId = range(7, 19)
	# featuresId = [12,14,16,17,18]
	# nbInput = len(featuresId)
	X, Y = generateDataset(d, featuresId)

	# print X
	# print "------------"
	# print Y
	# print "------------"

	X = featureExpand(d, X)
	X = featureRescale(d, X)

	#################################################
	# import matplotlib.pyplot as plt
	# from numpy.random import rand

	# fig, ax = plt.subplots()
	# for i,dt in enumerate(X):
	# 	plt.scatter(np.arange(len(X[i])), X[i], alpha=0.3, edgecolors='none')
	# # plt.scatter(np.arange(len(X[3])), X[3], alpha=0.3, edgecolors='none')

	# plt.show()

	#################################################
	allclassifier = MultiClassifier(nbInput, nbOutput)

	for i in range(nbOutput):
		allclassifier.addClassifier(i)

	lr = 1000.0
	oldLoss = 9e+9
	allclassifier.setLr(lr)

	allclassifier.printInfo()

	for j in range(20000):
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
