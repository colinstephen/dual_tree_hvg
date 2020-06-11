#!/usr/bin/env python
# -*- coding: UTF-8 -*-



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Colin Stephen
# DATE:  May 2020
# CONTACT: colin.stephen@coventry.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""
Runs numerical experiments to compare run times and base/recursive call counts
when constructing HVGs using different methods on different time series. 
"""


# Runtime experiments with various types of time series of various lengths


import numpy as np
import streams
from bst_hvg import hvg as binary_search_hvg
from dc_hvg import hvg as divide_conquer_hvg
from dt_hvg import hvg as dual_tree_hvg


# Set up the time series data sources

sources = {
	'random': streams.random,
	'random_walk': streams.discrete_random_walk,
	'logistic': streams.logistic_attractor,
	'henon': streams.henon_attractor
}


for hurst in np.linspace(0.1, 0.9, num=9):
	sources[f'fbm_{hurst}'] = partial(streams.fbm, hurst=hurst)


for tmax in [50, 100, 250]:
	sources[f'lorenz_{tmax}'] = partial(streams.lorenz_attractor, tmax=tmax)


for tmax in [500, 1000, 2500]:
	sources[f'rossler_{tmax}'] = partial(streams.rossler_attractor, tmax=tmax)


reps = range(1,11)
lengths = np.logspace(8, 17, num=10, base=2, dtype=np.int)

for source in sources:
	for rep in reps:
		for n in lengths:
			x = sources[source](n)

			pass

			# now run each of the HVG algorithms on x and time it
			# record times to an external results file