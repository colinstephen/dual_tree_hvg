#!/usr/bin/env python
# -*- coding: UTF-8 -*-



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Colin Stephen
# DATE:  June 2020
# CONTACT: colin.stephen@coventry.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""
Run numerical experiments to compare run times
of HVG construction using different algorithms. 
"""



# See https://pythonspeed.com/articles/python-multiprocessing/
# on why mp pools should spawn new processes rather than forking them
from multiprocessing import get_context
from multiprocessing import freeze_support

import sys
import csv
import time
import pickle
import numpy as np

from functools import partial
from itertools import repeat
from datetime import datetime

import streams
from bst_hvg import hvg as binary_search_hvg
from dc_hvg import hvg as divide_conquer_hvg
from dt_hvg import hvg as dual_tree_hvg

sys.setrecursionlimit(25000)  # needed for DC method
now = datetime.now



def generate_time_series(data_key, sources):
	'''
	Uses a source stream name and a time series length specified in `data_key`
	to generate a time series. A dictionary of sources `sources` provides access
	to the actual source function.
	'''

	source, length, rep = data_key  # NB: can ignore rep

	# Some sources may diverge, overflow, or exceed machine precision
	# so try multiple times to generate a sequence that works 
	# and return None if it fails every time.

	tries_remaining = 10
	time_series = None

	while time_series is None and tries_remaining > 0:
		try:
			time_series = sources[source](length)
		except Exception as e:
			tries -= 1

	return data_key, time_series



def time_algorithm(experiment_params, dict_of_algs, dict_of_time_series):
	'''
	Apply an HVG algorithm to data using one of the algs
	and return the elapsed thread/process time.
	'''

	alg, data_key = experiment_params
	hvg_algorithm = dict_of_algs[alg]
	time_series = dict_of_time_series[data_key]

	try:
		t0 = time.perf_counter()
		_ = hvg_algorithm(time_series)
		result = time.perf_counter() - t0
	except Exception as e:
		raise(e)
		# Return -1 when an algorithm crashes
		# (usually due to too much recursion).
		result = -1

	return experiment_params, result



def get_experiment(EXPERIMENT, TESTING=True):

	# ~~~~~~~~~~~~
	# EXPERIMENT 1
	# ~~~~~~~~~~~~

	# -----------------------------------------------------------------
	# MIX OF RANDOM AND CHAOTIC SYSTEMS - LENGTHS 2**8-2**16 - ALL ALGS
	# -----------------------------------------------------------------

	if EXPERIMENT == 1:
		
		sources = {
			'random': streams.random,
			'random_walk': streams.discrete_random_walk,
			'logistic': streams.logistic_attractor,
			'standard': streams.standard_map
		}

		tmaxs = [50, 100, 250]
		for tmax in tmaxs:
			sources[f'lorenz_{tmax}'] = partial(streams.lorenz_attractor,
				tmax=tmax)

		tmaxs = [2**x for x in range(8,12)]
		for tmax in tmaxs:
			sources[f'rossler_{tmax}'] = partial(streams.rossler_attractor,
				tmax=tmax)

		if TESTING:
			lengths = [2**n for n in range (8,10)]
			reps = range(2)
		else:
			lengths = [2**n for n in range(8,17)]
			reps = range(10)

		algs = {
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


	if EXPERIMENT == 2:

		sources = {}

		hursts = np.linspace(0.2, 0.8, num=13)
		for hurst in hursts:
			sources[f'fbm_{hurst:.2f}'] = partial(streams.fbm, hurst=hurst)

		slopes = np.linspace(0, 3, num=16)
		for m in slopes:
			sources[f'linear_trend_{m:.1f}'] = partial(streams.linear_trend, m=m)

		seasonal_frequencies = np.linspace(1, 4, num=16)
		for freq in seasonal_frequencies:
			sources[f'seasonal_frequency_{freq:.1f}'] = partial(streams.seasonal_trend,
				frequency=freq)

		seasonal_amplitudes = np.linspace(0, 3, num=16)
		for amp in seasonal_amplitudes:
			sources[f'seasonal_amplitude_{amp:.1f}'] = partial(streams.seasonal_trend,
				amplitude=amp)

		if TESTING:
			lengths = [2**8]
			reps = range(2)
		else:
			lengths = [2**17]
			reps = range(10)

		algs = {
			'dual_tree_hvg': dual_tree_hvg,
			'binary_search_hvg': binary_search_hvg,
		}


	assert EXPERIMENT in [1, 2]


	data_filename = f'experiment_{EXPERIMENT}{"_TESTING" if TESTING else ""}_data.pickle'
	results_filename = f'experiment_{EXPERIMENT}{"_TESTING" if TESTING else ""}_results.csv'


	return {
		'sources': sources,
		'lengths': lengths,
		'reps': reps,
		'algs': algs,
		'data_filename': data_filename,
		'results_filename': results_filename
	}



def run_experiment(EXPERIMENT, TESTING=True):

	print(f'Begin running experiment {EXPERIMENT}: {now()}')
	
	experiment = get_experiment(EXPERIMENT, TESTING=TESTING)

	sources = experiment['sources']
	lengths = experiment['lengths']
	reps = experiment['reps']
	algs = experiment['algs']
	data_filename = experiment['data_filename']
	results_filename = experiment['results_filename']

	# ~~~~~~~~~~~~~~~~~
	# GENERATE THE DATA
	# ~~~~~~~~~~~~~~~~~

	data = {}
	data_keys = [(source, length, rep) for source in sources for length in
		lengths for rep in reps]

	print(f'\tBegin generating experimental data: {now()}')

	pool = get_context("spawn").Pool()
	time_series_data = pool.starmap(generate_time_series, zip(data_keys,
		repeat(sources)), chunksize=1)

	for data_key, time_series in time_series_data:
		if time_series is None:
			print(f'\t\tgenerating data with parameters {data_key} failed')
		else:
			data[data_key] = time_series

	pool.close()
	pool.join()

	pickle.dump(data, open(data_filename, 'wb'))

	print(f'\tSaved experimental data to {data_filename}: {now()}')



	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Run the algorithms on the experimental data
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	exp_keys = [(alg, data_key) for data_key in data_keys for alg in algs]

	print(f'\tBegin running experiments: {now()}')

	pool = get_context("spawn").Pool()
	runtime_results = pool.starmap(time_algorithm, zip(exp_keys, repeat(algs),
		repeat(data)), chunksize=1)

	f = open(results_filename, 'w')
	writer = csv.writer(f)
	csv_headers = ['source', 'length', 'rep', 'hvg_algorithm',
		'time_in_seconds']
	writer.writerow(csv_headers)

	completed = 0
	total = len(exp_keys)

	for (alg, (source, length, rep)), result in runtime_results:

		writer.writerow([source, length, rep, alg, result])

		completed += 1
		if completed % 50 == 0:
			print(f'\t\tCompleted {completed} of {total} at {now()}')

	pool.close()
	pool.join()
	f.close()

	print(f'\tCompleted saving results to {results_filename}: {now()}')

	print(f'Completed running experiment {EXPERIMENT}: {now()}')
	

if __name__ == '__main__':

	freeze_support()
	experiment = int(sys.argv[1]) if len(sys.argv) > 1 else 1
	run_experiment(experiment)

