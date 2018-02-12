#!/usr/bin/python3

import sys

sys.path.append('Class')
from Dataset import *
from IOHelper import *

import numpy as np
import matplotlib.pyplot as plt

def addFeatureOnSubplot(index1, index2, d, indexSubplot, width):

	str_limit = 5
	color = ['#FF6633', '#FFB399', '#FF33FF', '#FFFF99', '#00B3E6', 
		  '#E6B333', '#3366E6', '#999966', '#99FF99', '#B34D4D',
		  '#80B300', '#809900', '#E6B3B3', '#6680B3', '#66991A', 
		  '#FF99E6', '#CCFF1A', '#FF1A66', '#E6331A', '#33FFCC',
		  '#66994D', '#B366CC', '#4D8000', '#B33300', '#CC80CC', 
		  '#66664D', '#991AFF', '#E666FF', '#4DB3FF', '#1AB399',
		  '#E666B3', '#33991A', '#CC9999', '#B3B31A', '#00E680', 
		  '#4D8066', '#809980', '#E6FF80', '#1AFF33', '#999933',
		  '#FF3380', '#CCCC00', '#66E64D', '#4D80CC', '#9900B3', 
		  '#E64D66', '#4DB380', '#FF4D4D', '#99E6E6', '#6666FF']

	X = d.getFeature(index1)
	Y = d.getFeature(index2)

	# X.sort()
	# Y.sort()
	len1 = len(X)
	len2 = len(Y)
	max1 = len1
	if len1 > len2:
		max1 = len1
	if len2 > len1:
		max1 = len2

	X.resize(max1)
	Y.resize(max1)

	ax = plt.subplot(width, width, indexSubplot)

	clr = color[1]
	plt.scatter(Y, X, color=clr, alpha=0.3)
	clr = color[6]
	plt.scatter(X, Y, color=clr, alpha=0.3)
	plt.tight_layout()

	frame1 = plt.gca()
	frame1.axes.xaxis.set_ticklabels([])
	frame1.axes.yaxis.set_ticklabels([])
	if (indexSubplot -1) % width == 0:
		plt.ylabel(d.getName(index1)[:str_limit])
	if (indexSubplot -1) % width != 0 and indexSubplot <= width:
		plt.title(d.getName(index2)[:str_limit])

def drawOneSub(d, Xstart, Ystart, range1):
	
	x = 0
	for i in range1:
		sys.stdout.write('.')
		sys.stdout.flush()
		addFeatureOnSubplot(Xstart, Ystart + x, d, i+1, len(range1))
		x += 1


def main():

	file = IOHelper().checkArg(sys.argv)
	if (len(file) < 1):
		print("Missing file")
		exit(1)

	d = Dataset()
	d.loadFile(file[0])

	fig, axes = plt.subplots(figsize=(18,10))
	fig.tight_layout()

	start = 6
	width = 13

	widthStart = 0
	widthEnd = widthStart + width
	ystart = start
	for i in range(width):
		drawOneSub(d, start, ystart, range(widthStart, widthEnd))
		widthStart += width
		widthEnd += width
		start += 1
	print("")

	# plt.title(d.getName(index))

	plt.savefig('scatter_plot.png')
	plt.show()


main()
