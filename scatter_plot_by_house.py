#!/usr/bin/python3

import sys

sys.path.append('Class')
from Dataset import *
from IOHelper import *

import numpy as np
import matplotlib.pyplot as plt

def main():

	file = IOHelper().checkArg(sys.argv)
	if (len(file) < 1):
		print("Missing file")
		exit(1)

	d = Dataset()
	d.loadFile(file[0])

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

		plt.scatter(np.arange(len(x1)), x1, c='b', s=10, alpha=0.3, label='Gryffindor')
		plt.scatter(np.arange(len(x2)), x2, c='g', s=10, alpha=0.3, label='Hufflepuff')
		plt.scatter(np.arange(len(x3)), x3, c='c', s=10, alpha=0.3, label='Ravenclaw')
		plt.scatter(np.arange(len(x4)), x4, c='r', s=10, alpha=0.3, label='Slytherin')

		plt.scatter(np.arange(len(x1)), x1s, c='b', s=10, alpha=0.3)
		plt.scatter(np.arange(len(x2)), x2s, c='g', s=10, alpha=0.3)
		plt.scatter(np.arange(len(x3)), x3s, c='c', s=10, alpha=0.3)
		plt.scatter(np.arange(len(x4)), x4s, c='r', s=10, alpha=0.3)

		plt.title(d.getName(index))
		plt.ylabel('Worst <---> Best')
		plt.xlabel('Evaluation')

		plt.legend()
		plt.tight_layout()
		# plt.set_xlim([-10, len(x1) + 10])
		plt.show()


main()
