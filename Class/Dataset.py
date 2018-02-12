#!/usr/bin/python3

import csv
import numpy as np
from Math import *

class Dataset(Math):

	_X = []
	_len_X = 0
	_Nm = []

	def __init__(self):
		pass

	################################## GET ##################################

	def getDataset(self, x=-1, y=-1):
		if self._len_X != None:

			if self._len_X -1 < x:
				print("Error when trying to get element in dataset at line " + str(x+2))
				exit(1)
			if len(self._X[x]) -1 < y:
				print("Error when trying to get element in dataset at line " + str(x+2) + ", index " + str(y))
				exit(1)

			if x == -1 and y == -1:
				return (self._X)
			if x != -1 and y == -1:
				return (self._X[x])
			if x != -1 and y != -1:
				return (self._X[x][y])
		else:
			print("Error when getting dataset")
			exit(1)

	def getLength(self):
		return (self._len_X)

	def getName(self, index):
		for x in self._Nm:
			if len(x[index]) > 0:
				return x[index]
		return None

	def getFeature(self, index, column=-1, name='', uniq=False):
		if index > self._len_X:
			print("Error: Out of bound trying to get element at index " + str(index))
			exit(1)

		X = []
		if column != -1:
			for x in self._X:
				if len(x[index]) > 0 and x[column] == name:
					X.append(float(x[index]))
		else:
			if self._isFloat(self.getDataset(0, index)) == True:
				for i,d in enumerate(self._X):
					if len(self.getDataset(i,index)) > 0:
						X.append(float(self.getDataset(i,index)))
			else:
				for i,d in enumerate(self._X):
					if len(self.getDataset(i,index)) > 0:
						X.append(self.getDataset(i,index))
		if uniq == True:
			return np.unique(X)
		else:
			return np.array(X)

	################################## INIT ##################################

	def loadFile(self, file):
		with open(file, "r") as file:
			arr = csv.reader(file, delimiter=',')
			for i,line in enumerate(arr):
				if i == 0:
					self._Nm.append(line)
					continue
				self._X.append(line)

		self._len_X = len(self._X)
		return (self._X, self._Nm)

	################################## CALC ##################################

	def rescaleCore(self, x, min, max):
		ret = 1.0*(((x - (min)) / (max - min)))
		return ret

	def meanNormalization(self, x, moy, min, max):
		ret = 1.0*(((x - (moy)) / (max - min)))+0.5
		return ret

	def standardization(self, x, moy, std):
		ret = 1.0*(((x - (moy)) / (std)))+0.5
		return ret

	def featureRescale(self, d, X):
		min = self.min2D(X)
		max = self.max2D(X)
		# moy = self.moy2D(X)
		# std = self.std2D(X)
		newX = []

		for i,data1 in enumerate(X):
			for j,data2 in enumerate(data1):
				X[i][j] = self.rescaleCore(data2, min, max)
				# X[i][j] = meanNormalization(data2, moy, min, max)
				# tmp = standardization(data2, moy, std)
				# X[i][j] = rescaleCore(tmp, -30, 30)
		return X

	def featureExpand(self, d, X):
		global nbInput
		newX = []

		for i,data1 in enumerate(X):
			tmp = []
			for j in data1:
				tmp.append(j)

			# for i in range(3):
			tmp.append(1) # intercept

			l = len(X[0])
			for k in range(l):
				# tmp.append(1) # intercept

				for j in range(l):
					pass
					# if j+1 != k:
						# tmp.append(data1[k]*data1[(j+1)%l])

			for i in range(5):
				tmp.append(1)

			nbInput = len(tmp)
			newX.append(tmp) 
		return np.array(newX), nbInput

	################################## PRINT ##################################

	def printFeatureHeader(self): 
		print("                                         Count           Std             Mean            Min             25%             50%             75%             Max")

	def printAllFeature(self):
		for i in range(len(self.getDataset(0))):
			self.printFeature(i)

	def printFeature(self, index):
		if self._isFloat(self.getDataset(0,index)) == False or self.getName(index) == "Index":
			return;
	
		nom = self.getName(index)
		nb = self.count(self.getFeature(index))
		moy = self.mean(self.getFeature(index))
		std = self.standardDeviation(self.getFeature(index), moy)
		min1 = self.min(self.getFeature(index))
		q25, q50, q75 = self.quartile(self.getFeature(index))
		max1 = self.max(self.getFeature(index))
	 
		print("{0:<40s} {1:<15.5g} {2:<15.5g} {3:<15.5g} {4:<15.5g} {5:<15.5g} {6:<15.5g} {7:<15.5g} {8:<15.5g}" \
			.format(nom, nb, std, moy, min1, q25, q50, q75, max1))

	################################## ELSE ##################################

	def _isFloat(self, string):
		if string != None or len(string) > 0:
			try:
				x = float(string)
				return(True)
			except ValueError:
				return(False)
		return(False)
