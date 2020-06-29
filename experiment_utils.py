#!/usr/bin/env python
# -*- coding: UTF-8 -*-



import csv
import time
import pickle
from datetime import datetime
from multiprocessing import Pool



now = datetime.now



def generate_time_series(meta):
	'''
	Given some setup parameters for experiments, generate relevant time series.
	'''

	func = meta['data']['source']['func']
	length = meta['data']['length']
	time_series = func(length)

	return time_series



def time_algorithm(meta):
	'''
	Apply an HVG algorithm to data using one of the algs
	and return the elapsed time.
	'''

	func = meta['algorithm']['func']
	time_series = meta['data']['time_series']

	t0 = time.perf_counter()
	_ = func(time_series)
	runtime = time.perf_counter() - t0

	return runtime



def build_experiment_dict(algs, sources, lengths, reps):
	'''
	Build the data structure for the time series and timings
	'''

	experiment_data = []

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
					experiment_data.append(experiment)

	return experiment_data



def populate_time_series(experiment_data):

	print(f'\tBegin generating time series: {now()}')

	with Pool() as pool:
		
		results = pool.imap(generate_time_series, experiment_data)
		
		completed = 0
		total = len(experiment_data)
			
		for exp in experiment_data:
			exp['data']['time_series'] = results.next()
		
			completed += 1
			if completed % 50 == 0:
				print(f'\t\tCompleted {completed} of {total}: {now()}')

		pool.close()
		pool.join()
	
	print(f'\tEnd generating time series: {now()}')
	


def record_timings(experiment_data, results_file, data_file):

	with open(results_file, 'w') as f:

		writer = csv.writer(f)
		writer.writerow(['source', 'length', 'rep', 'hvg_algorithm', 'time_in_seconds'])
	
		with Pool() as pool:

			print(f'\tBegin timing algorithms: {now()}')
			
			completed = 0
			total = len(experiment_data)
			
			results = pool.imap(time_algorithm, experiment_data)
			
			for exp in experiment_data:
				runtime = results.next()
				exp['result'] = runtime

				source = exp['data']['source']['name']
				length = exp['data']['length']
				rep = exp['data']['rep']
				alg = exp['algorithm']['name']

				writer.writerow([source, length, rep, alg, runtime])

				completed += 1
				if completed % 50 == 0:
					print(f'\t\tCompleted {completed} of {total}: {now()}')

			pool.close()
			pool.join()

	print(f'\tSaved results to {results_file}: {now()}')

	pickle.dump(experiment_data, open(data_file, 'wb'))

	print(f'\tSaved data to {data_file}: {now()}')


