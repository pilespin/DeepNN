#!/usr/bin/python

import numpy as np
import csv
import sys	
# import time
from Dataset import *
from Math import *

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

def sigmaTh(d, m, th1, th_J, houseArray):
	# sum sigma
	Xall = d.getDataset()

	i = 0
	sigma = 0.0
	while i < m:

		X = getLayer(d, i, inFloat=True)
		house = Xall[i][1]
		Y = getIndex(houseArray, house)
		if Y == 1:
			# Htheta = np.sum(predict(X, th1)) * X[th_J]
			# print predict(X, th1)
			Htheta = np.sum(predict(X, th1)) **2
			# Htheta = np.sum(predict(X, th1))
			tmp = Htheta
			# sigma = sigma + (Y * np.log(Htheta)) + (1 - Y) * np.log(1 - Htheta)
			sigma += tmp
			# print sigma
		i+=1
	# print sigma
	return sigma

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

def updateLr(th, loss, lr):
	if loss > 0:
		th = th - lr
	else:
		th = th + lr
	return th

##############################
############ MAIN ############
##############################

def main():

	file = checkArg(sys.argv)

	d = Dataset()
	ma = Math()

	d.loadFile(file)

	m = d.count(d.getFeature(0))

	lr = 0.01
	nbInput = 13
	houseArray = d.getFeature(1, uniq=True)
	print houseArray
	nbOutput = len(houseArray)
	print houseArray

	# th1 = np.array([0]*nbInput, dtype=float)
	th1 = np.zeros(nbInput, dtype=float)
	# print predict(X, th1)
	print th1

	for x in range(10000):
		x+=1
		sigma = np.array([0]*nbInput, dtype=float)
		loss = np.array([0]*nbInput, dtype=float)
		j = 0
		for i in th1:
			# sigma[j] = updateLr(th1[j], loss, lr)
			sigma[j] = sigmaTh(d, m, th1, j, houseArray)
			loss[j] = lr * (sigma[j] / m)
			j+=1

		print "Sigma: " + str(sigma)
		print " Loss: " + str(loss)
		print "TLoss: " + str(np.sum(loss))
		j = 0
		for i in th1:
			th1[j] = updateLr(th1[j], loss[j], lr)
			j+=1
		print "Theta: " + str(th1)
		index = 10
		X = getLayer(d, index, inFloat=True)
		print "Feature: " + str(X) + " House: " + str(d.getDataset()[index][1])
		tmp = np.sum(predict(X, th1))
		print "Predict: " + str(ma.sigmoid([tmp]))
		print "------------------------------------------"


	
main()
