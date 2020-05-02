#!/usr/bin/env python
# -*- coding: UTF-8 -*-



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Colin Stephen
# DATE:  April 2020
# CONTACT: colin.stephen@coventry.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""
Provides functions to create and consume streams of time series and convert them
in to horizontal visibility graphs (HVGs).
"""



import time
import numpy as np
from linear_hvg import batch_hvg as linear_hvg
from linear_hvg import merge as linear_merge
from linear_hvg import HVG
from binary_hvg import hvg as binary_hvg
from binary_hvg import merge as binary_merge


def random(batch_size=1, random_seed=None):
	np.random.seed(random_seed)
	while True:
		yield np.random.random(size=batch_size)



def normal(batch_size=1, random_seed=None):
	np.random.seed(random_seed)
	while True:
		yield np.random.normal(size=batch_size)



def random_walk(batch_size=1, random_seed=None):
    np.random.seed(random_seed)
    end = 0
    while True:
    	start = end + (-1 if np.random.random()<0.5 else 1)
    	out = [start]
        while(len(out) < batch_size):
	        out.append(out[-1] + (-1 if np.random.random()<0.5 else 1 ))
    	yield np.array(out)
    	end = out[-1]


def process_stream(hvg, source, batches=100, times=True):
	# Use the hvg_builder and hvg_merger to process a stream of batches of time
	# series values.
	#
	# The source is assumed to be a generator.
	
	total_time = 0

	# the first batch starts a new graph
	batch = next(source)
	
	t0 = time.time()
	hvg.add_batch(batch)
	total_time += time.time() - t0
	
	# subsequent batches are then used to extend the graph 
	for i, batch in zip(range(batches-1), source):
		t0 = time.time()
		# new_hvg = hvg_builder(batch)
		# hvg = hvg_merger(hvg, new_hvg)
		hvg.add_batch(batch)
		total_time += time.time() - t0

	if times:
		return hvg, total_time
	else:
		return hvg



random_seed = np.random.randint(0,100)
batches = 100000
batch_size = 10

source = random_walk(batch_size=batch_size, random_seed=random_seed)
total_time = 0
hvg = HVG()
for _, batch in zip(xrange(batches), source):
	t0 = time.time()
	hvg.add_batch(batch)
	total_time += time.time() - t0
print('linear time:', total_time)

source = random_walk(batch_size=batch_size, random_seed=random_seed)
total_time = 0
batch = next(source)
t0 = time.time()
hvg = binary_hvg(batch)
total_time += time.time() - t0
for _, batch in zip(xrange(batches-1), source):
	t0 = time.time()
	hvg_batch = binary_hvg(batch)
	hvg = binary_merge([hvg, hvg_batch])
	total_time += time.time() - t0
print('binary time:', total_time)

# source = random(batch_size=1000, random_seed=random_seed)
# _, linear_times = process_stream(linear_hvg, linear_merge, source)
# _, binary_times = process_stream(binary_hvg, binary_merge, source)

# source = random_walk(batch_size=2000000, random_seed=random_seed)
# _, linear_times = process_stream(linear_hvg, linear_merge, source, batches=1)
# print('linear time:', linear_times)
