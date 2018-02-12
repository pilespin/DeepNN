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
		tmp = []
		for i in line[1:]:
			tmp.append(float(i))

		X.append(tmp)
		Y.append(line[:1][0])
	
	if len(X) != len(Y):
		print("Error")
		exit(1)
	
	return (X, Y)

#############################################################

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
	if (len(file) < 2):
		print "Missing file"
		exit(1)

	d = Dataset()
	d.loadFile(file[0])

	featuresId = range(7, 19)
	# nbInput = len(featuresId)
	X = generateDataset(d, featuresId)

	X, nbInput = d.featureExpand(d, X)
	X = d.featureRescale(d, X)

	allWeight, AllOutput = csvToArray(file[1])

	allclassifier = MultiClassifier(nbInput, AllOutput)
	allclassifier.initWeight(allWeight)

	with open('houses.csv', 'w') as file:
		file.write("Index,Hogwarts House\n")
		for i,d in enumerate(X):
			name = allclassifier.predict(d)
			file.write(str(i) + "," + name + "\n")


main()
