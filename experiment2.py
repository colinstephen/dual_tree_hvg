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



import numpy as np
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

	print(f'Begin running experiment 2: {now()}')

	results_file = 'temp/experiment_2_results.csv'
	data_file = 'temp/experiment_2_data.pickle'

	algs = {
		'dual_tree_hvg': dual_tree_hvg,
		'binary_search_hvg': binary_search_hvg
	}

	sources = {}

	for hurst in np.linspace(0.2, 0.8, num=13):
		sources[f'fbm_{hurst:.2f}'] = partial(streams.fbm, hurst=hurst)

	for tmax in [2**n for n in range(8,15)]:
		sources[f'rossler_{tmax}'] = partial(streams.rossler_attractor, tmax=tmax)

	for noise_factor in range(0,65,2):
		sources[f'noisy_cycles_{noise_factor}'] = partial(streams.two_cycles,
			noise_factor=noise_factor)

	lengths = [2**15]

	reps = range(10)

	experiment_data = build_experiment_dict(algs, sources, lengths, reps)
	populate_time_series(experiment_data)
	record_timings(experiment_data, results_file, data_file)
	
	print(f'Completed running experiment 2: {now()}')

	

if __name__ == '__main__':

	main()
