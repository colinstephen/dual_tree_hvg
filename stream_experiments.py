import time
import numpy as np
from fbm import FBM
from streams import random_walk, fbm
from linear_hvg import HVG
from binary_hvg import hvg as binary_hvg

import sys
sys.setrecursionlimit(2500) # Binary HVG needs this for smooth correlated sequences

# batch_size = 100
# batches = 200
# random_seed = np.random.randint(0, 100)
# source = random_walk(batch_size=batch_size, random_seed=random_seed)
# hvg = HVG()
# for _, batch in zip(range(batches), source):
# 	hvg.add_batch(batch)

# batch_size = 2**13
# for i in range(5, 6):
# 	hurst = i/10
# 	reps = 5
# 	total_time = 0
# 	for rep in range(reps):
# 		f = FBM(batch_size, hurst=hurst)
# 		fbm_sample = f.fbm()
# 		t0 = time.time()
# 		hvg = binary_hvg(fbm_sample)
# 		total_time += time.time() - t0
# 	mean_time = total_time / reps
# 	print(f"hurst = {hurst} : mean time = {mean_time}")


batch_size = 2**13
total_time = 0
reps = 5
for rep in range(reps):
	sample = list(random_walk(batch_size=batch_size))
	print(sample)
	break
	t0 = time.time()
	hvg = binary_hvg(sample)
	total_time += time.time() - t0
mean_time = total_time / reps
print(f"random walk : mean time = {mean_time}")
