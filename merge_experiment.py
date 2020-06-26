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
sys.setrecursionlimit(25000)  # needed for search tree method

TESTING = True

# ~~~~~~~~~~~~~~
# Financial data
# ~~~~~~~~~~~~~~

filenames = [f'data/finance0{i}.csv.gz' for i in range(1,6)]
algs = {'dual_tree_hvg': dual_tree_hvg, 'binary_search_hvg': binary_search_hvg}
chunk_sizes = np.linspace(1000, 499999, num=50, dtype=int)

if TESTING:
	filenames = [f'data/finance0{i}_TESTING.csv.gz' for i in range(1,6)]
	chunk_sizes = np.linspace(32, 1023, num=5, dtype=int)

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

	data = np.loadtxt(filename)

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
	
	results = pool.imap(record_run_times, experimental_data)
	
	for exp in experimental_data:

		hvg_times, merge_times = results.next()
		exp['hvg_times'] = hvg_times
		exp['merge_times'] = merge_times

		completed += 1
		if completed % 50 == 0:
			print(f'\t\tCompleted {completed} of {total}: {now()}')

	pool.close()
	pool.join()



results_datafile = 'temp/merge_experiment_finance_results.pickle'
if TESTING:
	results_datafile = 'temp/merge_experiment_finance_results_TESTING.pickle'
with open(results_datafile, 'wb') as f:
	pickle.dump(experimental_data, f)
print(f'\tSaved finance result data to {results_datafile}: {now()}')


results_csvfile = 'temp/merge_experiment_finance_results.csv'
if TESTING:
	results_csvfile = 'temp/merge_experiment_finance_results_TESTING.csv'
with open(results_csvfile, 'w') as f:
	writer = csv.writer(f)
	writer.writerow(['filename', 'algorithm', 'chunk_size', 'hvg_total',
		'hvg_mean', 'hvg_median', 'hvg_std', 'merge_total','merge_mean',
		'merge_median', 'merge_std'])
	for exp in experimental_data:
		fn = exp['filename']
		alg = exp['algorithm']['name']
		chnk = exp['chunk_size']
		hvg_times = exp['hvg_times'][:-1]  # last HVG is a different length
		hvg_sum = np.sum(hvg_times)
		hvg_mean = np.mean(hvg_times)
		hvg_med = np.median(hvg_times)
		hvg_std = np.std(hvg_times)
		merge_times = exp['merge_times'][:-1]  # final merge is shorter
		merge_sum = np.sum(merge_times)
		merge_mean = np.mean(merge_times)
		merge_med = np.median(merge_times)
		merge_std = np.std(merge_times)
		writer.writerow([fn, alg, chnk, hvg_sum, hvg_mean, hvg_med, hvg_std,
			merge_sum, merge_mean, merge_med, merge_std])
print(f'\tSaved finance summary results to {results_csvfile}: {now()}')

