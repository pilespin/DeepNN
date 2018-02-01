#!/usr/bin/python

import csv
import sys
import math

from Dataset import *

import numpy as np
import matplotlib.pyplot as plt

import time
import sys

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

	# index = 11

	for index in range(6, 19):

		x1 = d.getFeature(index, 1, 'Gryffindor')
		x2 = d.getFeature(index, 1, 'Hufflepuff')
		x3 = d.getFeature(index, 1, 'Ravenclaw')
		x4 = d.getFeature(index, 1, 'Slytherin')

		x1s = sorted(d.getFeature(index, 1, 'Gryffindor'))
		x2s = sorted(d.getFeature(index, 1, 'Hufflepuff'))
		x3s = sorted(d.getFeature(index, 1, 'Ravenclaw'))
		x4s = sorted(d.getFeature(index, 1, 'Slytherin'))

		ax = plt.subplot(1, 1, 1)
		plt.tight_layout()
		ax.set_xlim([-10, len(x1) + 10])

		plt.scatter(np.arange(len(x1)), x1, c='b', alpha=0.5, label='Gryffindor')
		plt.scatter(np.arange(len(x2)), x2, c='g', alpha=0.5, label='Hufflepuff')
		plt.scatter(np.arange(len(x3)), x3, c='c', alpha=0.5, label='Ravenclaw')
		plt.scatter(np.arange(len(x4)), x4, c='r', alpha=0.5, label='Slytherin')

		plt.scatter(np.arange(len(x1)), x1s, c='b', alpha=0.5)
		plt.scatter(np.arange(len(x2)), x2s, c='g', alpha=0.5)
		plt.scatter(np.arange(len(x3)), x3s, c='c', alpha=0.5)
		plt.scatter(np.arange(len(x4)), x4s, c='r', alpha=0.5)

		plt.title(d.getName(index))
		plt.ylabel('Worst <---> Best')
		plt.xlabel('Evaluation')

		plt.legend()
		plt.tight_layout()
		# plt.set_xlim([-10, len(x1) + 10])
		plt.show()



main()
