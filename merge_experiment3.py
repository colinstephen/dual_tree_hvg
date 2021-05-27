from multiprocessing import set_start_method
set_start_method("spawn")

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

import streams

now = datetime.datetime.now
sys.setrecursionlimit(25000)  # needed for search tree method

TESTING = True

# ~~~~~~~~
# FBM Data
# ~~~~~~~~

def setup_data():
	algs = {'dual_tree_hvg': dual_tree_hvg, 'binary_search_hvg': binary_search_hvg}

	if TESTING:
		reps = 1
		hurst_exponents = np.linspace(0.2, 0.8, num=3)
		chunk_sizes = range(2**5, 2**9 + 1, 2**5)
		data_length = 2**10
	else:
		reps = 3
		hurst_exponents = np.linspace(0.2, 0.8, num=13)
		chunk_sizes = range(2**12, 2**18 + 1, 2**12)
		data_length = 2**20

	experimental_data = []

	for rep in range(reps):
		print(f'Generating data for rep {rep} at {now()}')
		with Pool() as pool:
			fbm_data = list(pool.imap(streams.fbm, hurst_exponents))
			# pool.close()
			# pool.join()
		for i, hurst in enumerate(hurst_exponents):
			data = fbm_data[i]
			for chunk_size in chunk_sizes:
				for alg in algs:
					experimental_data += [{
						'rep': rep,
						'hurst_exponent': hurst,
						'algorithm': {'name': alg, 'function': algs[alg]},
						'chunk_size': chunk_size,
						'hvg_times': None,
						'merge_times': None,
						'data': data
					}]

	return experimental_data

def record_run_times(exp):

	hvg_func = exp['algorithm']['function']
	chunk_size = exp['chunk_size']
	data = exp['data']
	hvg_times = []
	merge_times = []

	# compute the small HVGs to be merged
	hvgs = []
	for n in range(0, len(data), chunk_size):
		t0 = time.perf_counter()
		hvg = hvg_func(data[n:n+chunk_size])
		hvg_times += [time.perf_counter() - t0]
		hvgs += [hvg]

	# then repeatedly merge them together
	merged = hvgs[0]
	for hvg in hvgs[1:]:
		t0 = time.perf_counter()
		merged += hvg
		merge_times += [time.perf_counter() - t0]

	return hvg_times, merge_times

def run_experiments(experimental_data):
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

		# pool.close()
		# pool.join()

def save_outputs(experimental_data):
	if TESTING:
		results_datafile = 'temp/merge_experiment_fbm_results_TESTING.pickle'
	else:
		results_datafile = 'temp/merge_experiment_fbm_results.pickle'

	# forget the data before saving as we can just re-generate
	for instance in experimental_data:
		del instance['data']

	with open(results_datafile, 'wb') as f:
		pickle.dump(experimental_data, f)

	print(f'\tSaved fbm merge result data to {results_datafile}: {now()}')

	if TESTING:
		results_csvfile = 'temp/merge_experiment_fbm_results_TESTING.csv'
	else:
		results_csvfile = 'temp/merge_experiment_fbm_results.csv'

	with open(results_csvfile, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(['rep', 'hurst_exponent', 'algorithm', 'chunk_size', 'hvg_total',
			'hvg_mean', 'hvg_median', 'hvg_std', 'merge_total','merge_mean',
			'merge_median', 'merge_std'])
		for exp in experimental_data:
			rep = exp['rep']
			hurst_exponent = exp['hurst_exponent']
			algorithm = exp['algorithm']['name']
			chunk_size = exp['chunk_size']
			hvg_times = exp['hvg_times']
			hvg_sum = np.sum(hvg_times)
			hvg_mean = np.mean(hvg_times)
			hvg_median = np.median(hvg_times)
			hvg_std = np.std(hvg_times)
			merge_times = exp['merge_times']
			merge_sum = np.sum(merge_times)
			merge_mean = np.mean(merge_times)
			merge_median = np.median(merge_times)
			merge_std = np.std(merge_times)
			writer.writerow([rep, hurst_exponent, algorithm, chunk_size, hvg_sum,
				hvg_mean, hvg_median, hvg_std, merge_sum, merge_mean, merge_median,
				merge_std])

	print(f'\tSaved fbm summary results to {results_csvfile}: {now()}')

if __name__ == '__main__':
	data = setup_data()
	run_experiments(data)
	save_outputs(data)
