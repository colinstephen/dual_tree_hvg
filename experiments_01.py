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



TESTING = True
data_filename = 'experiments_01_data.pickle'
results_filename = r'experiments_01_results.csv'



import csv
import time
import pickle
import numpy as np
import multiprocessing as mp

from functools import partial
from datetime import datetime
now = datetime.now

import streams
from bst_hvg import hvg as binary_search_hvg
from dc_hvg import hvg as divide_conquer_hvg
from dt_hvg import hvg as dual_tree_hvg

import sys
sys.setrecursionlimit(25000)  # needed for DC method



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate the time series data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sources = {
	'random': streams.random,
	'random_walk': streams.discrete_random_walk,
	'logistic': streams.logistic_attractor,
	'standard': streams.standard_map
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


if TESTING:
	lengths = [2**x for x in range(8,10)]
	reps = range(3)
else:
	lengths = [2**x for x in range(8,17)]
	reps = range(11)


data_keys = [(source, length, rep) for source in sources for length in lengths
	for rep in reps]


def generate_time_series(data_key):
	source, length, rep = data_key  # can ignore rep

	# some sources may diverge, overflow, or exceed machine precision
	# so try multiple times to generate a sequence that works 

	tries_remaining = 10
	time_series = None

	while time_series is None and tries_remaining > 0:
		try:
			time_series = sources[source](length)
		except Exception as e:
			tries -= 1

	return time_series


print(f'Begin generating experimental data at {now()}')


pool = mp.Pool()
time_series_data = pool.imap(generate_time_series, data_keys)
pool.close()
pool.join()


data = {}
for data_key in data_keys:
	time_series = time_series_data.next()
	if time_series is None:
		print(f'generating data with parameters {data_key} failed')
	else:
		data[data_key] = time_series
pickle.dump(data, open(data_filename, 'wb'))


print(f'Saved experimental data to {data_filename} at {now()}')



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the algorithms on the experimental data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

algs = {
	'dual_tree_hvg': dual_tree_hvg,
	'binary_search_hvg': binary_search_hvg,
	'divide_conquer_hvg': divide_conquer_hvg
}


exp_keys = [(alg, data_key) for data_key in data_keys for alg in algs]


def time_algorithm(experiment_params):
	'''
	Apply an HVG algorithm to data and return the elapsed thread/process time.

	'''

	alg, data_key = experiment_params
	hvg_algorithm = algs[alg]
	time_series = data[data_key]

	try:
		t0 = time.perf_counter()
		_ = hvg_algorithm(time_series)
		result = time.perf_counter() - t0
	except Exception as e:
		# Return -1 when an algorithm crashes
		# (usually due to too much recursion).
		result = -1

	return result


print(f'Begin running experiments at {now()}')


# The experiments take a wide range of durations to complete
# so set the pooling chunk size to 1 here to maximise resource usage.
pool = mp.Pool()
runtime_results = pool.imap(time_algorithm, exp_keys, chunksize=1)


with open(results_filename, 'w') as f:

	writer = csv.writer(f)
	csv_headers = ['source', 'length', 'rep', 'hvg_algorithm', 'time_in_seconds']
	writer.writerow(csv_headers)

	completed = 0
	total = len(exp_keys)

	for alg, (source, length, rep) in exp_keys:

		runtime_result = runtime_results.next()
		writer.writerow([source, length, rep, alg, runtime_result])

		completed += 1
		if completed % 50 == 0:
			print(f'Completed {completed} of {total} at {now()}')

	print(f'Completed saving results to {results_filename} at {now()}')


pool.close()
pool.join()



