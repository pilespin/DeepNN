#!/usr/bin/python

import numpy as np
import csv
import sys	
# import time
from Dataset import *
from Classifier import *
from Math import *
# from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

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
			# print int(i)
			return int(i)
	return -1

def getInputInDataset(d, index, inFloat=False):
	start = 6
	end = 18

	X = np.array([])
	if inFloat == True:
		while start <= end:
			tmp = d.getDataset()[index][start]
			if len(tmp) > 0:
				X = np.append(X, float(d.getDataset()[index][start]))
			else:
				X = np.append(X, float(0))
			start += 1
	else:
		while start <= end:
			X = np.append(X, d.getDataset()[index][start])
			start += 1
	return X

def generateDataset(d, index=-1):

	X = []
	Y = []

	houseArray = d.getFeature(1, uniq=True)

	for i in range(d.getLength()):
		x = getInputInDataset(d, i, inFloat=True)
		y = getIndex(houseArray, getHouseByIndex(d, i))

		if index == -1 or (y == index):
			# print y
			X.append(x)
			Y.append([y])

	X = np.array(X)
	Y = np.array(Y)
	if len(X) != len(Y):
		print "Error when generate dataset"
		exit(1)
	return X, Y

def generatePrediction(c, X, Y):
	y_pred = []
	y_true = []

	for i,x in enumerate(X):
		output = c.predict(x)
		y_pred.append([output])
		y_true = Y[i]
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
	c = []
	for i in range(nbOutput):
		c.append(Classifier(nbInput, nbOutput))
	# ma = Math()
	d.loadFile(file)

	index = 1
	houseArray = d.getFeature(1, uniq=True)
	Xtest = getInputInDataset(d, index, inFloat=True)
	Ytest = getIndex(houseArray, getHouseByIndex(d, index))

	X = [None]*nbOutput
	Y = [None]*nbOutput

	for i in range(nbOutput):
		X[i], Y[i] = generateDataset(d, index=i+1)
		# print X[i]
		# print Y[i]

	print "train"
	for i in range(1000):
		for i in range(nbOutput):
			# pass
			# print i
			c[i].train(X[i], Y[i])
			output = c[i].predict(Xtest)
			print "class " + str(i) + " -- " + str(m.sigmoid_core(output.sum()))
			# y_true, y_pred = generatePrediction(c, X, Y)
			# accuracy_score(y_true, y_pred)
		print "-----------------"


main()
