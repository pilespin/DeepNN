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


	print d.getName(1)
	for house in d.getFeature(1, uniq=True):
		print house
		d.printFeatureHeader()
		index = 6
		# i = 6
		while index <= 18:
			nom = d.getName(index)
			nb = d.count(d.getFeature(index, 1, house))
			moy = d.mean(d.getFeature(index, 1, house))
			std = d.standardDeviation(d.getFeature(index, 1, house), moy)
			min1 = d.min(d.getFeature(index, 1, house))
			q25, q50, q75 = d.quartile(d.getFeature(index, 1, house))
			max1 = d.max(d.getFeature(index, 1, house))

			print("{0:<40s} {1:<15.5g} {2:<15.5g} {3:<15.5g} {4:<15.5g} {5:<15.5g} {6:<15.5g} {7:<15.5g} {8:<15.5g}" \
				.format(nom, nb, std, moy, min1, q25, q50, q75, max1))
			index+=1
		print ""

main()
