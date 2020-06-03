#!/usr/bin/env python
# -*- coding: UTF-8 -*-



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Colin Stephen
# DATE:  April 2020
# CONTACT: colin.stephen@coventry.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""
Provides functions to construct horizontal visibility graphs (HVGs) of time
series directly, or via online streaming of batched time series data of
arbitrary length. Works in worst-case O(n) time.
"""



import numpy as np
from scipy.sparse import dok_matrix



class HVG:
  '''
  Class for the weighted horizontal visibility of a sequence.
  '''


  def __init__(self):
    self.X = []  # the actual time series
    self.length = 0  # update to avoid multiple calls to `len(self.X)`
    self.E = []  # list of weighted HVG edges [u, v, w]
    self.vis_p = []  # longest strictly increasing subsequence from self.X[0]
    self.vis_f = []  # longest strictly decreasing subsequence to self.X[-1] 
    self.max_val = -np.inf  # update to avoid multiple calls to `max(self.X)`
    self.neighbour_weight = np.inf  # value of w for edges [u, u+1, w]
    self._A = None  # upper triangular adjacency matrix corresponding to self.E


  def __len__(self):
    return self.length


  def add_one(self, x):

    v = self.length
    self.X += [x]
    self.length += 1
    # we now have self.X[v] = x
        
    if x > self.max_val:
      self.vis_p += [v]
      self.max_val = x

    while self.vis_f:
      
      u = self.vis_f[-1]
      
      if self.X[u] <= self.X[v]:
        if v == u+1:
          self.E += [[u, v, self.neighbour_weight]]
        else:
          self.E += [[u, v, self.X[u]-h]]
        del self.vis_f[-1]
        if self.X[u] == self.X[v]:
          break
        h = self.X[u]

      else:
        if v == u+1:
          self.E += [[u, v, self.neighbour_weight]]
        else:
          self.E += [[u, v, self.X[v]-h]]
        break

    self.vis_f += [v]

    return self


  def add_batch(self, batch):
    for x in batch:
      self.add_one(x)

    return self


  @property
  def A(self):
    if self._A is None:
      A = dok_matrix((self.length, self.length), dtype=np.float32)
      for u, v, w in self.E:
        A[u,v] = w
      self._A = A
    return self._A


  def merge(self, other, copy=True):
    '''
    Merge an `other` HVG to the right of the current HVG
    
    By default returns a new HVG object.
    
    Setting `copy=False` can be useful for multiple iterated merges when the
      intermediate results are not important.
    '''

    # When using weights in merges the neighbour weights must match.
    assert self.neighbour_weight == other.neighbour_weight

    # Compute the adjacency for the joint graph
    L1, L2 = self.length, other.length
    N = L1 + L2
    A = dok_matrix((N, N), dtype=np.float32)

    # We already have the upper left and lower right blocks.
    A[:L1, :L1] = self.A
    A[L1:, L1:] = other.A

    # But some nodes from the two graphs may be visible to one another.
    keys = self.vis_f + [v + L1 for v in other.vis_p]
    vals = [self.X[v] for v in self.vis_f] + [other.X[v] for v in other.vis_p]

    # We treat the possibly visible values as a subsequence and create its HVG.
    hvg = HVG().add_batch(vals)

    # Its edges give the off diagonal blocks in the joint adjacency matrix.
    for u, v, w in hvg.E:
      # We know u is in the current HVG and v is in the `other` HVG.
      # Moreover u and v are indirectly indexed via the list of `keys`.
      A[keys[u], keys[v]] = w

    # Compute the combined time series and the new past/future node indices
    X = self.X + other.X
    vis_p = self.vis_p + [v + L1 for v in other.vis_p if v > self.max_val]
    vis_f = [v for v in self.vis_f if v > other.max_val]
    vis_f += [v + L1 for v in other.vis_f]

    if copy:
      hvg = HVG()
    else:
      hvg = self

    hvg.X = X
    hvg._A = A
    hvg.length = N
    hvg.vis_p = vis_p
    hvg.vis_f = vis_f
    
    # Reset the edge list as we don't need it now and it is slow to work with.
    # Prefer the adjacency representation when merges are involved.
    hvg.E = []

    return hvg

