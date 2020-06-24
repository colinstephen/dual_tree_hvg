import os
import sys
import time
import datetime
import numpy as np
from dt_hvg import hvg as dual_tree_hvg
now = datetime.datetime.now

if os.environ.get('TESTING') == 'False':
	TESTING = False
else:
	TESTING = True
print(f'Running experiment with TESTING=={TESTING}')

print(f'starting: {now()}')
data = []
with open('data/finance01.csv', 'r') as f:
	for line in f:
		data += [float(line.strip())]
		if TESTING and len(data) >= 2**18:
			break
print(f'data in memory: {now()}')

# print(f'Starting baseline HVG timing: {now()}')
# t0 = time.perf_counter()
# _ = dual_tree_hvg(data)
# t1 = time.perf_counter()
# print(f'Baseline computation time is {t1-t0} seconds - completed at: {now()}')

if TESTING:
	chunk_size = 2048
else:
	chunk_size = 2048

hvgs = [dual_tree_hvg(data[n:n+chunk_size]) for n in range(0, len(data), chunk_size)]
print(f'hvgs computed: {now()}')

merge_result = hvgs[0]

times = []
for hvg in hvgs[1:]:
	t0 = time.perf_counter()
	merge_result = merge_result + hvg
	# merge_result += hvg
	times += [time.perf_counter() - t0]
	if len(times) % 50 == 0:
		print(f'\tcompleted {len(times)} merges of {len(hvgs)-1}')

print(f'total time {np.sum(times)} and mean merge time {np.mean(times)}')
sys.exit()

def generate_hvgs(file_name, chunk_size, hvg_alg):

	print(f'loading data from {file_name}: {now()}')
	data = np.loadtxt(file_name)  # only a few million lines so load it all
	print(f'completed data load: {now()}')
	N = len(data)
	if TESTING:
		N = 40960
	chunk_indexes = range(0, N, chunk_size)
	num_chunks = len(chunk_indexes)
	print(f'total length of data: {N}')
	print(f'will generate {num_chunks} HVGs of length {chunk_size}')
	return [hvg_alg(data[n:n+chunk_size]) for n in chunk_indexes]

	# for n in chunk_indexes:
	# 	x = data[n:n+chunk_size]
	# 	hvg = hvg_alg(x)
	# 	if 'A' in dir(hvg):
	# 		_ = hvg.A
	# 	yield hvg



def time_merges(hvgs):

	merge_times = []
	merged_hvg = hvgs[0]  # no merging needed for the first one
	_ = merged_hvg.A  # ensure adjacency is available
	
	for hvg in hvgs[1:]:
		# time subsequent additions
		_ = hvg.A
		t0 = time.perf_counter()
		merged_hvg = merged_hvg + hvg
		merge_times += [time.perf_counter() - t0]
	
		if len(merge_times) % 50 == 0:
			print(f'\tprocessed {len(merge_times)} HVGs: {now()}')

	return merge_times



if __name__ == '__main__':
	hvgs = generate_hvgs('data/finance01.csv', 2**10, dual_tree_hvg)
	times = time_merges(hvgs)
	print(np.mean(times))


