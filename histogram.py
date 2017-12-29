#!/usr/bin/python

import csv
import sys
import math

from Dataset import *

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

def main():

	file = checkArg(sys.argv)

	d = Dataset()

	d.loadFile(file)

	# d.printFeatureHeader()
	# d.printAllFeature()

	# print d.count(d.getFeature(16, 1, "Gryffindor"))
	# print d.count(d.getFeature(16, 1, "Gryffindor"))
	print d.getName(1)
	for house in d.getFeature(1, uniq=True):
		print house
		print d.count(d.getFeature(16, 1, house))
		

main()
