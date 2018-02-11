#!/usr/bin/python3

import numpy as np
import csv
import sys

sys.path.append('Class')
from Dataset import *
from MultiClassifier import *
from IOHelper import *


# np.set_printoptions(precision=4)
np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)

def csvToArray(file):
	file = open(file, "r")
	arr = csv.reader(file, delimiter=',')

	X = []
	Y = []

	for i,line in enumerate(arr):
		X.append(line[1:])
		Y.append(line[:1])
	
	if len(X) != len(Y):
		print("Error")
		exit(1)
	
	return (X, Y)

#############################################################

# def getHouseByIndex(d, index):
# 	house = d.getDataset()[index][1]
# 	return house

# def getIndex(X, querie):
# 	for i,x in enumerate(X):
# 		if x == querie:
# 			return int(i+1)
# 	return -1

def getInputInDataset(d, index, featuresId, inFloat=False):
	global nbInput

	X = []
	if inFloat == True:
		for i in featuresId:
			# tmp = d.getDataset()[index][i]
			tmp = d.getDataset(index, i)
			if len(tmp) > 0:
				X.append(float(tmp))
			else:
				return None
				# X.append(float(1))
	else:
		for i in featuresId:
			X.append(d.getDataset(index, i))
			# X.append(d.getDataset()[index][i])

	return np.array(X)

def generateDataset(d, featuresId, index=-1):

	X = []

	houseArray = d.getFeature(1, uniq=True)

	for i in range(d.getLength()):
		x = getInputInDataset(d, i, featuresId, inFloat=True)

		if x is not None: 
			if index == -1 or (y == index):
				if len(x) > 0:
					X.append(x)

	X = np.array(X)
	if len(X) <= 0:
		print("Error Empty dataset")
		exit(1)
	return X

##############################
############ MAIN ############
##############################

def main():

	nbOutput = 4

	file = IOHelper().checkArg(sys.argv)

	d = Dataset()
	d.loadFile(file)

	featuresId = range(7, 19)
	nbInput = len(featuresId)
	X = generateDataset(d, featuresId)

	# print(X)

	allWweight, AllOutput = csvToArray("weight.csv")
	print X
	print Y

	# allclassifier = MultiClassifier(nbInput, nbOutput)

	# for i in range(nbOutput):
		# allclassifier.addClassifier(i)

	# loss = allclassifier.train(X, Y)


main()
