#!/usr/bin/env python
# -*- coding: UTF-8 -*-



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Colin Stephen
# DATE:  June 2020
# CONTACT: colin.stephen@coventry.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""
Run numerical experiments to compare HVG construction run times.

The experimental data are arranged in the following dictionary structure:
	{
		'algorithm': {'name': alg_name, 'func': alg_func},
		'data': {'source': {'name': source_name, 'func': source_func}
				 'length': length,
				 'rep': rep,
				 'time_series': None},
		'result': None
	}
"""



from functools import partial
from datetime import datetime

import streams
from bst_hvg import hvg as binary_search_hvg
from dc_hvg import hvg as divide_conquer_hvg
from dt_hvg import hvg as dual_tree_hvg

from experiment_utils import generate_time_series
from experiment_utils import time_algorithm
from experiment_utils import build_experiment_dict
from experiment_utils import populate_time_series
from experiment_utils import record_timings

now = datetime.now



def main():

	print(f'Begin running experiment 1: {now()}')

	results_file = 'temp/experiment_1_results.csv'
	data_file = 'temp/experiment_1_data.pickle'

	algs = {
		'dual_tree_hvg': dual_tree_hvg,
		'binary_search_hvg': binary_search_hvg,
		'divide_conquer_hvg': divide_conquer_hvg
	}

	sources = {
		'random': streams.random,
		'random_walk': streams.discrete_random_walk,
		'logistic': streams.logistic_attractor,
		'standard': streams.standard_map,
		'lorenz_250': partial(streams.lorenz_attractor, tmax=250),
		'rossler_1024': partial(streams.rossler_attractor, tmax=1024)
	}

	lengths = [2**n for n in range(8,17)]

	reps = range(10)

	experiment_data = build_experiment_dict(algs, sources, lengths, reps)
	populate_time_series(experiment_data)
	record_timings(experiment_data, results_file, data_file)
	
	print(f'Completed running experiment 1: {now()}')

	

if __name__ == '__main__':

	main()
