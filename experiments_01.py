#!/usr/bin/env python
# -*- coding: UTF-8 -*-



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Colin Stephen
# DATE:  May 2020
# CONTACT: colin.stephen@coventry.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""
Run numerical experiments to compare run times
of HVG construction using different algorithms. 
"""



import csv
import time
import numpy as np
from functools import partial

import streams
from bst_hvg import hvg as binary_search_hvg
from dc_hvg import hvg as divide_conquer_hvg
from dt_hvg import hvg as dual_tree_hvg

import sys
sys.setrecursionlimit(25000)  # needed for DC method



# Set up the time series data sources

sources = {
	'random': streams.random,
	'random_walk': streams.discrete_random_walk,
	'logistic': streams.logistic_attractor,
	'henon': streams.henon_attractor
}

hursts = np.linspace(0.15, 0.85, num=15)
for hurst in hursts:
	sources[f'fbm_{hurst:.2f}'] = partial(streams.fbm, hurst=hurst)


tmaxs = [50, 100, 250]
for tmax in tmaxs:
	sources[f'lorenz_{tmax}'] = partial(streams.lorenz_attractor, tmax=tmax)


tmaxs = [2**x for x in range(8,12)]
for tmax in tmaxs:
	sources[f'rossler_{tmax}'] = partial(streams.rossler_attractor, tmax=tmax)



# Set up the experimental parameters

reps = range(10)
lengths = [2**x for x in range(8,18)]
results_file = r'results_01.csv'
csv_headers = ['source_name', 'length', 'bst_time', 'dc_time', 'dt_time']



# Run the experiments

def time_algorithm(alg, x):
	'''
	Apply algorithm to time series x and return the elapsed process time.
	Return -1 when the algorithm crashes, usually due to too much recursion.
	'''
	t0 = time.process_time()
	try:
		_ = alg(x)
		t1 = time.process_time()
	except Exception as e:
		t1 = t0 - 1
	return t1 - t0


with open(results_file, 'w') as f:

	writer = csv.writer(f)
	writer.writerow(csv_headers)

	for source in sources:

		for n in lengths:

			for rep in reps:

				x = sources[source](n)

				# run each of the HVG algorithms on x and time it
				# record times to the results csv file

				t0 = time_algorithm(binary_search_hvg, x)
				t1 = time_algorithm(divide_conquer_hvg, x)
				t2 = time_algorithm(dual_tree_hvg, x)

				writer.writerow([source, n, t0, t1, t2])
