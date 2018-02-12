#!/usr/bin/python3

from sys import path
path.append('Class')
from Dataset import *
from IOHelper import *

def main():

	file = IOHelper().checkArg(sys.argv)
	if (len(file) < 1):
		print "Missing file"
		exit(1)

	d = Dataset()
	d.loadFile(file[0])

	d.printFeatureHeader()
	d.printAllFeature()


main()
