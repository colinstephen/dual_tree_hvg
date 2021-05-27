#!/usr/bin/env python

TESTING = True

import os
import ipyparallel as ipp

slurm_profile_is_available = os.path.exists(os.path.expanduser('~/.ipython/profile_slurm/'))
c = ipp.Client(profile="slurm" if slurm_profile_is_available else "default")
v = c[:]

import sys
import csv
import datetime
import numpy as np

from dt_hvg import hvg as dual_tree_hvg
from bst_hvg import hvg as binary_search_hvg
sys.setrecursionlimit(25000)  # needed for bst method to work on unbalanced data

now = datetime.datetime.now
print(f'Beginning FBM merge experiment: {now()}')
print(f'Number of parallel tasks: {len(v)}')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up experimental parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

hvg_algorithms = {
	'dual_tree_hvg': dual_tree_hvg,
	'binary_search_hvg': binary_search_hvg
}

if TESTING:
	reps = 1
	hurst_exponents = [0.2, 0.5, 0.8] * reps
	chunk_sizes = range(2**5, 2**9 + 1, 2**5)
	data_length = 2**10
else:
	reps = 3
	hurst_exponents = list(np.linspace(0.2, 0.8, num=13)) * reps
	chunk_sizes = range(2**12, 2**18 + 1, 2**12)
	data_length = 2**20

# ~~~~~~~~~~~~~~~~~
# Generate the data
# ~~~~~~~~~~~~~~~~~

print(f'Beginning data generation: {now()}')

import streams
@ipp.require(streams.fbm, data_length=data_length)
def get_data_parallel(hurst):
	return streams.fbm(data_length, hurst=hurst)

fbm_data = v.map_sync(get_data_parallel, hurst_exponents)

experimental_data = [
	{
		'hurst_exponent': hurst,
		'data': data,
		'chunk_size': chunk,
		'algorithm': alg
	}
	for hurst, data in zip(hurst_exponents, fbm_data)
	for alg in hvg_algorithms
	for chunk in chunk_sizes
]

print(f'Completed data generation: {now()}')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parallel function to time HVG merging
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@ipp.require(hvg_algorithms=hvg_algorithms)
def record_run_times_parallel(experiment_params):
	import time
	
	data = experiment_params['data']
	chunk_size = experiment_params['chunk_size']
	hvg_algorithm = hvg_algorithms[experiment_params['algorithm']]

	hvg_times = []
	merge_times = []

	# compute the small HVGs to be merged
	hvgs = []
	for n in range(0, len(data), chunk_size):
		t0 = time.perf_counter()
		hvg = hvg_algorithm(data[n:n+chunk_size])
		hvg_times += [time.perf_counter() - t0]
		hvgs += [hvg]

	# then repeatedly merge them together
	merged = hvgs[0]
	for hvg in hvgs[1:]:
		t0 = time.perf_counter()
		merged += hvg
		merge_times += [time.perf_counter() - t0]

	return hvg_times, merge_times, experiment_params

# ~~~~~~~~~~~~~~~~~~
# Record the results
# ~~~~~~~~~~~~~~~~~~

print(f'Beginning timings: {now()}')

timings_async = v.map(record_run_times_parallel, experimental_data)

if TESTING:
	results_csvfile = 'temp/merge_experiment_fbm_results_TESTING.csv'
else:
	results_csvfile = 'temp/merge_experiment_fbm_results.csv'

with open(results_csvfile, 'w') as f:
	writer = csv.writer(f)
	writer.writerow([
		'hurst_exponent',
		'algorithm',
		'chunk_size',
		'hvg_total',
		'hvg_mean',
		'hvg_median',
		'hvg_std',
		'merge_total',
		'merge_mean',
		'merge_median',
		'merge_std'
	])

	count = 0
	total = len(experimental_data)
	for hvg_times, merge_times, experiment_params in timings_async:
		writer.writerow([
			experiment_params['hurst_exponent'],
			experiment_params['algorithm'],
			experiment_params['chunk_size'],
			np.sum(hvg_times),
			np.mean(hvg_times),
			np.median(hvg_times),
			np.std(hvg_times),
			np.sum(merge_times),
			np.mean(merge_times),
			np.median(merge_times),
			np.std(merge_times)
		])
		count +=1
		if count % 25 == 0:
			print(f'Completed {count} of {total} timings: {now()}')

print(f'Completed all timings: {now()}')
