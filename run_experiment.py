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

# See https://pythonspeed.com/articles/python-multiprocessing/
# on why mp pools should spawn new processes rather than forking them
# from multiprocessing import get_context
from multiprocessing import Pool



def generate_time_series(experiment_data):
	'''
	Uses a source stream name and a time series length specified in `data_key`
	to generate a time series. A dictionary of sources `sources` provides access
	to the actual source function.

	The argument `experiment_data` is a dict:
		{
			'algorithm': {'name': alg_name, 'func': alg_func},
			'data': {'source': {'name': source_name, 'func': source_func}
					 'length': length,
					 'rep': rep,
					 'time_series': None},
			'result': None
		}
	
	The job of this function is to populate the value for
	experiment_data['data']['time_series'] as a sequence of values.

	Some sources may diverge, overflow, or exceed machine precision
	so try multiple times to generate a sequence that works 
	and return None if it fails every time.
	'''
	time_series_data = experiment_data['data']

	func = time_series_data['source']['func']
	length = time_series_data['length']

	tries_remaining = 10
	time_series = None

	while time_series is None and tries_remaining > 0:
		try:
			time_series = func(length)
		except Exception as e:
			tries -= 1

	return time_series

	# time_series_data['time_series'] = time_series

	# return time_series_data



def time_algorithm(experiment_data):
	'''
	Apply an HVG algorithm to data using one of the algs
	and return the elapsed thread/process time.
	
	The argument `experiment_data` is a dict:
		{
			'algorithm': {'name': alg_name, 'func': alg_func},
			'data': {'source': {'name': source_name, 'func': source_func}
					 'length': length,
					 'rep': rep,
					 'time_series': sequence_of_values},
			'result': None
		}
	
	The job of this function is to populate the value for
	experiment_data['result'] as a duration in seconds.
	'''
	func = experiment_data['algorithm']['func']
	time_series = experiment_data['data']['time_series']

	try:
		t0 = time.perf_counter()
		_ = func(time_series)
		result = time.perf_counter() - t0
	except Exception as e:
		# Return -1 when an algorithm crashes
		# (usually due to too much recursion).
		result = -1

	return result

	# experiment_data['result'] = result

	# return experiment_data



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
			lengths = [2**12]
			reps = range(2)
		else:
			lengths = [2**17]
			reps = range(10)

		algs = {
			'dual_tree_hvg': dual_tree_hvg,
			'binary_search_hvg': binary_search_hvg,
		}


	assert EXPERIMENT in [1, 2]


	results_filename = f'experiment_{EXPERIMENT}{"_TESTING" if TESTING else ""}_results.csv'


	experimental_data = []

	'''
	Now build a list of dictionaries of the following form:
		{
			'algorithm': {'name': alg_name, 'func': alg_func},
			'data': {'source': {'name': source_name, 'func': source_func},
					 'length': length,
					 'rep': rep,
					 'time_series': None},
			'result': None
		}
	'''

	for alg in algs:
		algorithm = {'name':alg, 'func':algs[alg]}
		for source in sources:
			data_source = {'name':source, 'func':sources[source]}
			for length in lengths:
				for rep in reps:
					experiment = {
						'algorithm': algorithm,
						'data': {'source': data_source,
								 'length': length,
								 'rep': rep,
								 'time_series': None},
						'result': None
					}
					experimental_data.append(experiment)

	return experimental_data


def get_data_filename(EXPERIMENT, TESTING=True):
	return f'experiment_{EXPERIMENT}{"_TESTING" if TESTING else ""}_data.pickle'



def get_results_filename(EXPERIMENT, TESTING=True):
	return f'experiment_{EXPERIMENT}{"_TESTING" if TESTING else ""}_results.csv'



def save_result(result, file_handle):
	writer = csv.writer(file_handle)
	source = result['data']['source']['name']
	length = result['data']['length']
	rep = result['data']['rep']
	alg = result['algorithm']['name']
	result = result['result']
	writer.writerow([source, length, rep, alg, result])
	return None



def get_results_file(file_name):
	'''
	Get a file handle for a new results CSV file and populate the header row.
	'''

	f = open(file_name, 'w')
	writer = csv.writer(f)
	writer.writerow(['source', 'length', 'rep', 'hvg_algorithm',
		'time_in_seconds'])
	return f



def run_experiment(EXPERIMENT, TESTING=True):
	'''
	The main method of the script. Given an experiment number go through the
	following processes:

	* Generate the experimental parameters - algorithms, sequence lengths, etc.
	* Use the experiment parameters to generate time series data
		- runs in parallel
		- pickles the generated data to disk
	* Process the time series data with the selected algorithms and time them
		- runs in parallel
		- writes timing results to a CSV file as it goes
	'''

	data_filename = get_data_filename(EXPERIMENT, TESTING)
	results_filename = get_results_filename(EXPERIMENT, TESTING)


	print(f'Begin running experiment {EXPERIMENT}: {now()}')
	experimental_data = get_experiment(EXPERIMENT, TESTING=TESTING)


	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Generate the time series data
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	print(f'\tBegin generating experimental data: {now()}')

	with Pool() as pool:
		
		results = pool.imap_unordered(generate_time_series, experimental_data)
		
		for exp in experimental_data:
			exp['data']['time_series'] = results.next()
		
		pool.close()
		pool.join()
	
	print(f'\tEnd generating experimental data: {now()}')
	
	pickle.dump(experimental_data, open(data_filename, 'wb'))
	print(f'\tSaved time series data to {data_filename}: {now()}')


	
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Run the algorithms on the experimental data and save run times
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	results_file = get_results_file(results_filename)
	
	with Pool() as pool:
		print(f'\tBegin running experiments: {now()}')
		
		completed = 0
		total = len(experimental_data)
		
		results = pool.imap_unordered(time_algorithm, experimental_data)
		
		for exp in experimental_data:
			result = results.next()
			exp['result'] = result
			save_result(exp, results_file)

			completed += 1
			if completed % 50 == 0:
				print(f'\t\tCompleted {completed} of {total}: {now()}')

		pool.close()
		pool.join()

	results_file.close()
	print(f'\tSaved results to {results_filename}: {now()}')

	pickle.dump(experimental_data, open(data_filename, 'wb'))
	print(f'\tSaved all experiments and results to {data_filename}: {now()}')

	print(f'Completed running experiment {EXPERIMENT}: {now()}')

	

if __name__ == '__main__':

	experiment = int(sys.argv[1]) if len(sys.argv) > 1 else 1
	run_experiment(experiment)

