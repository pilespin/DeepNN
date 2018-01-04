#!/usr/bin/python

import numpy as np
import csv
import sys	
# import time
from Dataset import *
from Math import *

import matplotlib.pyplot as plt

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

	X = [] # Mileage
	Y = [] # Price

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

def sigmaTh(d, m, th1, houseArray):
	# sum sigma
	Xall = d.getDataset()
	# m = d.count(d.getFeature(0))

	# m = d.countByIndex(0)
	i = 0
	sigma = 0.0
	while i < m:
		X = getLayer(d, i, inFloat=True)
		house = Xall[i][1]
		Y = getIndex(houseArray, house)
		print "INDEX OF: " + str(Y)
		Htheta = np.sum(predict(X, th1))
		sigma = sigma + (Y * np.log(Htheta)) + (1 - Y) * np.log(1 - Htheta)
		i+=1
	return sigma

##############################
############ MAIN ############
##############################

def getLayer(d, index, inFloat=False):
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

def predict(X, th1):
	m = Math()
	return m.sigmoid(X*th1)

def getIndex(X, querie):
	i = 0
	for x in X:
		i+=1
		if x == querie:
			return i
	return -1


def main():

	file = checkArg(sys.argv)

	d = Dataset()
	m = Math()

	d.loadFile(file)

	m = d.count(d.getFeature(0))


	# start = 6
	# end = 18

	nbInput = 13
	houseArray = d.getFeature(1, uniq=True)
	nbOutput = len(houseArray)
	print houseArray
	# print outArray.where('Hufflepuff')
	# in0 = np.zeros(nbInput, dtype=float)
	# out0 = np.zeros(nbOutput, dtype=float)
	# print in0
	# print out0
	# X = getLayer(d, 0, start, end, inFloat=True)	# Input
	# Y = getLayer(d, 0, 1, 1)						# Output
	# print X
	# print "----------------"
	# print Y

	th1 = np.array([2]*nbInput, dtype=float)
	# print predict(X, th1)
	print th1
	sig = sigmaTh(d, m, th1, houseArray)

	loss = sig / m
	print sig
	print loss
	# in1 = np.ones(nbInput, dtype=float)
	# out1 = np.zeros(nbOutput, dtype=float)
	# print "----------------"
	# print th1
	# print "----------------"
	# # print in1
	# # print "----------------"
	# print out1
	# print predict(X, th1)
	# print X*th1
	# print m.sigmoid(X*th1)

main()

