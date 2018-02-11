#!/usr/bin/python3

import sys

class IOHelper(object):

	def __init__(self):
		pass

	def checkArg(self, argv):
		if len(sys.argv) <= 1:
			print ("Missing file")
			exit(1)

		file = sys.argv[1]

		try:
			open(file, 'r')
		except IOError:
			print("Can't read: " + file)
			exit(1)
		return (file)
