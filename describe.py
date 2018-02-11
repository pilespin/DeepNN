#!/usr/bin/python3

import csv
import sys
import math

from Dataset import *
from IOHelper import *

def main():

	file = IOHelper().checkArg(sys.argv)

	d = Dataset()

	d.loadFile(file)

	d.printFeatureHeader()
	d.printAllFeature()


main()
