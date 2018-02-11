#!/usr/bin/python

# python -m cProfile -o stat.txt script.py

import pstats
p = pstats.Stats('stat.txt')
p.sort_stats('cumulative').print_stats(30)
