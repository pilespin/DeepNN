#!/usr/bin/python3

import sys

class IOHelper(object):

	def __init__(self):
		pass

	def checkOpenFile(self, file):
		try:
			open(file, 'r')
		except IOError:
			print("Can't read: " + file)
			exit(1)
		return (True)


	def checkArg(self, argv):
		if len(sys.argv) <= 1:
			print ("Missing file")
			exit(1)

		all = []
		for i,d in enumerate(sys.argv):
			if i != 0:
				if self.checkOpenFile(d) == True:
					all.append(d)

		return (all)
