#!/usr/bin/python

import time
from threading import Thread

class Cll(object):

	
	def __init__(self):
		pass

	def one(self, i):
	    print "sleeping 5 sec from thread %d" % i
	    time.sleep(5)
	    print "finished sleeping from thread %d" % i


# def myfunc(i):
#     print "sleeping 5 sec from thread %d" % i
#     time.sleep(5)
#     print "finished sleeping from thread %d" % i

d = Cll()

for i in range(10):
    t = Thread(target=d.one, args=(i,))
    t.start()