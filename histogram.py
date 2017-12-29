#!/usr/bin/python

import csv
import sys
import math

from Dataset import *

import numpy as np
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

def main():

	file = checkArg(sys.argv)

	d = Dataset()

	d.loadFile(file)


	print d.getName(1)
	HOUSE = {}
	for house in d.getFeature(1, uniq=True):
		HOUSE[house] = {}
		HOUSE[house]['1moy'] = []
		HOUSE[house]['2min'] = []
		HOUSE[house]['3q25'] = []
		HOUSE[house]['4q50'] = []
		HOUSE[house]['5q75'] = []
		HOUSE[house]['6max'] = []
		HOUSE['name'] = []
		print house
		d.printFeatureHeader()
		index = 6
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
			index += 1

			HOUSE[house]['1moy'].append(moy)
			HOUSE[house]['2min'].append(min1)
			HOUSE[house]['3q25'].append(q25)
			HOUSE[house]['4q50'].append(q50)
			HOUSE[house]['5q75'].append(q75)
			HOUSE[house]['6max'].append(max1)
			HOUSE['name'].append(nom)
		print ""
	
	x = np.arange(len(HOUSE['Gryffindor']['1moy']))

	width = 0.035
	w=0
	for i in sorted(HOUSE['Gryffindor'].iterkeys()):
		if w == 0:
			plt.bar(x + (width*(w+1)), HOUSE['Gryffindor'][i], width, color='b', label='Gryffindor')
			plt.bar(x + (width*(w+2)), HOUSE['Hufflepuff'][i], width, color='r', label='Hufflepuff')
			plt.bar(x + (width*(w+3)), HOUSE['Ravenclaw'][i], width, color='g', label='Ravenclaw')
			plt.bar(x + (width*(w+4)), HOUSE['Slytherin'][i], width, color='c', label='Slytherin')
		else:
			plt.bar(x + (width*(w+1)), HOUSE['Gryffindor'][i], width, color='b')
			plt.bar(x + (width*(w+2)), HOUSE['Hufflepuff'][i], width, color='r')
			plt.bar(x + (width*(w+3)), HOUSE['Ravenclaw'][i], width, color='g')
			plt.bar(x + (width*(w+4)), HOUSE['Slytherin'][i], width, color='c')
		w+=4
	
	plt.xticks(x, HOUSE['name'], rotation='vertical')
	
	plt.xlabel('Cours')
	plt.ylabel('Evaluation')

	plt.legend()
	plt.show()


main()
