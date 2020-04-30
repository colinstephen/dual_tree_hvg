# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Delia Fano Yela
# DATE:  May 2018
# CONTACT: d.fanoyela@qmul.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Script to test different series and save the different
# visibility algorithms computation times
# ------------------------------------------------------------------------------
# IMPORT
# ------------------------------------------------------------------------------
# For the methods :
from linear_hvg import linear_hvg
from binary_hvg import binary_hvg
from dc_hvg import dc_hvg
import numpy as np
# For the data generation:
import random
# For performance measurement:
import time


# ------------------------------------------------------------------------------
# RECURSION LIMIT
# ------------------------------------------------------------------------------
import sys
sys.setrecursionlimit(2500) # You might need this for DC algorithm

# ------------------------------------------------------------------------------
# SERIES
# ------------------------------------------------------------------------------
def randomwalk(N):
    # N : size of the series to be generated
    out = [-1 if random.random()<0.5 else 1]
    while(len(out) < N):
        out += [out[-1] + (-1 if random.random()<0.5 else 1 )]
    return np.array(out)


def conway(N):
    A = {1: 1, 2: 1}
    c = 1 #counter
    while N not in A.keys():
        if c not in A.keys():
            A[c] = A[A[c-1]] + A[c-A[c-1]]
        c += 1
    t = xrange(L)
    out = np.array([A.items()[x][1] for x in t]) - np.array(t)/2
    return out

# ------------------------------------------------------------------------------
# INITS
# ------------------------------------------------------------------------------

# all_series = ['walk', 'random','conway']
all_series = ['walk', 'random']

file = open("results_exp01.txt", "a+") # Open file to write results on
file.write('"computation_time","series_size","series_type","visibility","Method"\n ' )

for L in [10, 50, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5]:
# for L in [1e5,1e6,1e7]:
    print "L : ", L
    L = int(L)

    # ------------------------------------------------------------------------------
    # COMPUTATION TIME GRAPH FOR DIFFERENT SERIES
    # ------------------------------------------------------------------------------

    for series_type in all_series:

        # Conway series
        if series_type == 'conway':
            s = conway(L)
            S = 1 # the conway series will always take the same values, no point in repeating it
        else:
            S = 1


        for times in xrange(S):

            if series_type == 'random':
                s = np.random.random(L)
            elif series_type == 'walk':
                s = randomwalk(L)
            elif series_type == 'conway':
                pass

            timeLine = range(L)

            #"--------------------------------------"
            #" HORIZONTAL VISIBILITY GRAPH"
            #"--------------------------------------"

            #"DC HVG:"
            start = time.time()
            out = dc_hvg(s, 0, L)
            end1 = time.time()

            file.write("%.5f," %(end1 - start))
            file.write("%.5f," %L)
            file.write('"%s",' %series_type)
            file.write('"hvg",')
            file.write('"dc"\n')

            #"Binary HVG:"
            # start = time.time()
            # out = binary_hvg(s)
            # end1 = time.time()

            # file.write("%.5f," %(end1 - start))
            # file.write("%.5f," %L)
            # file.write('"%s",' %series_type)
            # file.write('"hvg",')
            # file.write('"bt"\n')

            #"Linear HVG:"
            # start = time.time()
            # out = linear_hvg(s)
            # end1 = time.time()

            # file.write("%.5f," %(end1 - start))
            # file.write("%.5f," %L)
            # file.write('"%s",' %series_type)
            # file.write('"hvg",')
            # file.write('"linear"\n')



file.close()
