#!/usr/bin/env python
# -*- coding: UTF-8 -*-



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Colin Stephen
# DATE:  May 2020
# CONTACT: colin.stephen@coventry.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""
Run numerical experiments to compare run times
of HVG construction using different algorithms. 
"""



EXPERIMENT = 2  # run this experiment
TESTING = True  # testing mode reduces computation time
data_filename = f'experiment_{EXPERIMENT}{"_TESTING" if TESTING else ""}_data.pickle'
results_filename = f'experiment_{EXPERIMENT}{"_TESTING" if TESTING else ""}_results.csv'



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



# ~~~~~~~~~~~~
# EXPERIMENT 1
# ~~~~~~~~~~~~

# -----------------------------------------------------------------
# MIX OF RANDOM AND CHAOTIC SYSTEMS - LENGTHS 2**8-2**16 - ALL ALGS
# -----------------------------------------------------------------

exp1_sources = {
	'random': streams.random,
	'random_walk': streams.discrete_random_walk,
	'logistic': streams.logistic_attractor,
	'standard': streams.standard_map
}

tmaxs = [50, 100, 250]
for tmax in tmaxs:
	exp1_sources[f'lorenz_{tmax}'] = partial(streams.lorenz_attractor,
		tmax=tmax)

tmaxs = [2**x for x in range(8,12)]
for tmax in tmaxs:
	exp1_sources[f'rossler_{tmax}'] = partial(streams.rossler_attractor,
		tmax=tmax)

if TESTING:
	exp1_lengths = [2**n for n in range (8,10)]
	exp1_reps = range(2)
else:
	exp1_lengths = [2**n for n in range(8,17)]
	exp1_reps = range(10)

exp1_algs = {
	'dual_tree_hvg': dual_tree_hvg,
	'binary_search_hvg': binary_search_hvg,
	'divide_conquer_hvg': divide_conquer_hvg
}



# ~~~~~~~~~~~~
# EXPERIMENT 2
# ~~~~~~~~~~~~

# -----------------------------------------------------------------
# TRENDED AND CORRELATED SEQUENCES - FIXED LENGTH 2**17 - FAST ALGS
# -----------------------------------------------------------------

exp2_sources = {}

hursts = np.linspace(0.2, 0.8, num=13)
for hurst in hursts:
	exp2_sources[f'fbm_{hurst:.2f}'] = partial(streams.fbm, hurst=hurst)

slopes = np.linspace(0, 2, num=11)
for m in slopes:
	exp2_sources[f'linear_trend_{m:.1f}'] = partial(streams.linear_trend, m=m)

seasonal_frequencies = np.linspace(1, 3, num=11)
for freq in seasonal_frequencies:
	exp2_sources[f'seasonal_frequency_{freq:.1f}'] = partial(streams.seasonal_trend,
		frequency=freq)

seasonal_amplitudes = np.linspace(0, 3, num=11)
for amp in seasonal_amplitudes:
	exp2_sources[f'seasonal_amplitude_{amp:.2f}'] = partial(streams.seasonal_trend,
		amplitude=amp)

if TESTING:
	exp2_lengths = [2**8]
	exp2_reps = range(2)
else:
	exp2_lengths = [2**17]
	exp2_reps = range(10)

exp2_algs = {
	'dual_tree_hvg': dual_tree_hvg,
	'binary_search_hvg': binary_search_hvg,
}



# ~~~~~~~~~~~
# EXPERIMENTS
# ~~~~~~~~~~~

experiments = {
	1: dict(sources=exp1_sources, lengths=exp1_lengths, reps=exp1_reps, algs=exp1_algs),
	2: dict(sources=exp2_sources, lengths=exp2_lengths, reps=exp2_reps, algs=exp2_algs),
}

experiment = experiments[EXPERIMENT]
sources = experiment['sources']
lengths = experiment['lengths']
reps = experiment['reps']
algs = experiment['algs']



# ~~~~~~~~~~~~~~~~~
# GENERATE THE DATA
# ~~~~~~~~~~~~~~~~~

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

	return data_key, time_series

data = {}
data_keys = [(source, length, rep) for source in sources for length in lengths
	for rep in reps]

print(f'Begin generating experimental data at {now()}')

pool = mp.Pool()
time_series_data = pool.imap_unordered(generate_time_series, data_keys)

for data_key, time_series in time_series_data:
	if time_series is None:
		print(f'generating data with parameters {data_key} failed')
	else:
		data[data_key] = time_series

pool.close()
pool.join()

pickle.dump(data, open(data_filename, 'wb'))

print(f'Saved experimental data to {data_filename} at {now()}')



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the algorithms on the experimental data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

	return experiment_params, result

exp_keys = [(alg, data_key) for data_key in data_keys for alg in algs]

print(f'Begin running experiments at {now()}')

pool = mp.Pool()
runtime_results = pool.imap_unordered(time_algorithm, exp_keys, chunksize=1)

f = open(results_filename, 'w')
writer = csv.writer(f)
csv_headers = ['source', 'length', 'rep', 'hvg_algorithm', 'time_in_seconds']
writer.writerow(csv_headers)

completed = 0
total = len(exp_keys)

for (alg, (source, length, rep)), result in runtime_results:

	writer.writerow([source, length, rep, alg, result])

	completed += 1
	if completed % 50 == 0:
		print(f'Completed {completed} of {total} at {now()}')

pool.close()
pool.join()
f.close()

print(f'Completed saving results to {results_filename} at {now()}')

