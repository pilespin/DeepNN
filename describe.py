#!/usr/bin/python

import numpy as np
import csv
import sys
import math

import matplotlib.pyplot as plt

def csvToArray(file):
	file = open(file, "r")
	arr = csv.reader(file, delimiter=',')

	X = []
	name = []

	i = 0
	for line in arr:
		if i == 0:
			i+=1
			name.append(line)
			continue
		X.append(line)
	return (X, name)

def count(X, index):
	i = 0
	for x in X:
		if len(x[index]) > 0:
			i+=1
	return i

def mean(X, index):
	s = 0
	i = 0
	for x in X:
		if len(x[index]) > 0:
			i+=1
			add = float(x[index])
			s += (add - s) / i
	return s

def standardDeviation(X, index, mean):
	s = 0
	i = 0
	for x in X:
		if len(x[index]) > 0:
			i+=1
			add = math.pow((float(x[index]) - mean), 2)
			s += (add - s) / i
	return math.sqrt(s)

def min(X, index):
	m = None
	for x in X:
		if len(x[index]) > 0:
			new = float(x[index])
			if (m == None):
				m = new
			elif new < m:
				m = new
	return m

def max(X, index):
	m = None
	for x in X:
		if len(x[index]) > 0:
			new = float(x[index])
			if (m == None):
				m = new
			elif new > m:
				m = new
	return m

def name(X, index):
	for x in X:
		if len(x[index]) > 0:
			return x[index]
	return None

def printFeatureHeader(): 
	print "                                         Count           Std             Mean            Min             25%             50%             75%             Max"

def printFeature(X, Nm, index):
	nom = name(Nm, index)
	nb = count(X, index)
	moy = mean(X, index)
	std = standardDeviation(X, index, moy)
	min1 = min(X, index)
	q25, q50, q75 = quartile(X, index)
	max1 = max(X, index)
 
	print("{0:<40s} {1:<15.5g} {2:<15.5g} {3:<15.5g} {4:<15.5g} {5:<15.5g} {6:<15.5g} {7:<15.5g} {8:<15.5g}" \
		.format(nom, nb, std, moy, min1, q25, q50, q75, max1))


##############################
############ MAIN ############
##############################

def medianArray(X):
	m = len(X)
	if m <= 0:
		print "Error when getting quartile array size of " + str(m)	
		exit(1)
	if m == 1:
		return [X[0]], [X[0]], X[0]
	if m % 2 == 0:
		# print "IS PAIR"
		first = m / 2
		second = m / 2
		med = (X[first]-1 + X[second]) / 2.0
		a = X[:first]
		b = X[second:m]
		return a, med, b
	else:
		# print "IS IMPAIR"
		first = ((m+1) / 2) - 1
		second = ((m+1) / 2)
		med = (X[first])
		a = X[:first]
		b = X[second:m]
		return a, med, b

def quartile(X1, index):
	X = []
	for x in X1:
		if len(x[index]) > 0:
			X.append(float(x[index]))

	X.sort()
	A, med, B = medianArray(X)
	n1, one, n2 = medianArray(A)
	n1, two, n2 = medianArray(B)

	return one, float(med), two

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

def isFloat(string):
	try:
		x = float(string)
		return(True)
	except ValueError:
		return(False)
	return(False)

def main():
	file = checkArg(sys.argv)

	X, Nm = csvToArray(file)

	printFeatureHeader()
	for i in range(len(X[0])):

		if isFloat(X[0][i]) == True:
			if name(Nm, i) != "Index":
				printFeature(X, Nm, i)

main()
