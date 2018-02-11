#!/usr/bin/python3

from sys import path
path.append('Class')
from Dataset import *
from IOHelper import *

def main():

	file = IOHelper().checkArg(sys.argv)

	d = Dataset()

	d.loadFile(file)

	d.printFeatureHeader()
	d.printAllFeature()


main()
