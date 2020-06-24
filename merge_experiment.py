import os
import sys
import csv
import pickle
import time
import datetime
import numpy as np
from multiprocessing import Pool

from dt_hvg import hvg as dual_tree_hvg
from bst_hvg import hvg as binary_search_hvg

now = datetime.datetime.now

TESTING = True

# ~~~~~~~~~~~~~~
# Financial data
# ~~~~~~~~~~~~~~

filenames = [f'data/finance0{i}.csv' for i in range(1,6)]
algs = {'dual_tree_hvg': dual_tree_hvg, 'binary_search_hvg': binary_search_hvg}
chunk_sizes = np.linspace(1000, 500000, num=50, dtype=int)

if TESTING:
	filenames = [f'data/finance0{i}_TESTING.csv' for i in range(1,6)]
	chunk_sizes = np.linspace(32, 1024, num=5, dtype=int)

experimental_data = [
	{
		'filename': filename,
		'algorithm': {'name': alg, 'function': algs[alg]},
		'chunk_size': chunk_size,
		'hvg_times': None,
		'merge_times': None,
		'merge_time_sum': -1,
		'merge_time_std': -1
	}
	for filename in filenames for alg in algs for chunk_size in chunk_sizes
]



def record_run_times(exp):

	filename = exp['filename']
	hvg_func = exp['algorithm']['function']
	chunk_size = exp['chunk_size']
	hvg_times = []
	merge_times = []

	data = []
	with open(filename, 'r') as f:
		for line in f:
			data += [float(line.strip())]
	data = np.array(data)

	hvgs = []
	for n in range(0, len(data), chunk_size):
		t0 = time.perf_counter()
		hvgs += [hvg_func(data[n:n+chunk_size])]
		hvg_times += [time.perf_counter() - t0]

	merged = hvgs[0]
	for hvg in hvgs[1:]:
		t0 = time.perf_counter()
		merged = merged + hvg
		merge_times += [time.perf_counter() - t0]

	return hvg_times, merge_times



with Pool() as pool:
	print(f'\tBegin running experiments: {now()}')
	
	completed = 0
	total = len(experimental_data)
	
	results = pool.imap_unordered(record_run_times, experimental_data)
	
	for exp in experimental_data:

		hvg_times, merge_times = results.next()
		exp['hvg_times'] = hvg_times
		exp['merge_times'] = merge_times
		exp['merge_time_sum'] = np.sum(merge_times)
		exp['merge_time_std'] = np.std(merge_times)

		completed += 1
		if completed % 50 == 0:
			print(f'\t\tCompleted {completed} of {total}: {now()}')

	pool.close()
	pool.join()



results_filename = 'temp/merge_experiment_fiance_results.pickle'
if TESTING:
	results_filename = 'temp/merge_experiment_finance_results_TESTING.pickle'
with open(results_filename, 'wb') as f:
	pickle.dump(experimental_data, f)

print(f'\tSaved finance results to {results_filename}: {now()}')


