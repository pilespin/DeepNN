#!/usr/bin/python

import numpy as np
import csv

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
			# print x[index]
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
		# print x
		if len(x[index]) > 0:
			return x[index]
	return None

def printFeatureHeader(): 
	print "                                         Count           Mean            Min             Max"

def printFeature(X, Nm, index):
	nom = name(Nm, index)
	nb = count(X, index)
	moy = mean(X, index)
	min1 = min(X, index)
	max1 = max(X, index)
 
	print("{0:<40s} {1:<15.5g} {2:<15.5g} {3:<15.5g} {4:<15.5g}" \
		.format(nom, nb, moy, min1, max1))


##############################
############ MAIN ############
##############################

def main():

	X, Nm = csvToArray("dataset_short.csv")

	printFeatureHeader()
	for i in range(len(X[0]) - 6):
		printFeature(X, Nm, i + 6)

main()
