#!/usr/bin/env python
# -*- coding: UTF-8 -*-



"""
AUTHOR: Colin Stephen
DATE:  April 2020
CONTACT: colin.stephen@coventry.ac.uk

DESCRIPTION

Provides a class to construct horizontal visibility graphs (HVGs) of time
series directly, or via online streaming of batch/packet time series data of
arbitrary size, and to add (merge) existing HVG graphs quickly.
"""



import numpy as np
from scipy.sparse import dok_matrix



class HVG:
  '''
  Class for the weighted horizontal visibility graph (HVG) of a sequence.

  METHODS
  - add_one() : build HVG incrementally with new values 
  - add_batch() : call add_one() multiple times on a list of values
  
  OPERATORS
  - + and += : efficient structural merge of existing HVGs
  
  PROPERTIES
  - A : Weighted upper triangular adjacency matrix
  '''


  def __init__(self, X_init=None):
    '''
    Sets up internal machinery for building HVGs from sequences.

    Optionally takes an initial sequence `X_init` to build the graph from.
    '''

    self.X = []  # the underlying time series sequence values
    self.vis_p = []  # longest strictly increasing subsequence from `self.X[0]`
    self.vis_f = []  # longest strictly decreasing subsequence to `self.X[-1]`
    self.max_val = -np.inf  # update to avoid multiple calls to `max(self.X)`
    self.neighbour_weight = np.inf  # value of `w` for edges `[u, u+1, w]`
    self._E = []  # list of weighted HVG edges `[u, v, w]`
    self._A = None  # upper triangular adjacency matrix of this graph

    # If a sequence was supplied, use it to build the graph.
    if X_init is not None and len(X_init):
      self.add_batch(X)

    return self


  def __len__(self):
    '''
    The length of an HVG is the number of its vertices.
    '''

    return len(self.X)


  def __add__(self, other):
    '''
    Allow adding HVGs via expressions like `hvg3 = hvg1 + hvg2`.
    '''

    return self.merge(other, copy=True)


  def __iadd__(self, other):
    '''
    Allow extending HVGs via expressions like `hvg1 += hvg2`
    '''

    return self.merge(other, copy=False)


  def add_one(self, x):
    '''
    Extend this HVG by adding edges induced by a new value `x`.
    '''

    v = len(self.X)
    self.X += [x]
    # We now have `self.X[v] == x`.
        
    if x > self.max_val:
      # Update the longest strictly increasing subsequence from `self.X[0]`.
      self.vis_p += [v]
      self.max_val = x

    while self.vis_f:
      # There is a longest strictly decreasing subsequence to `self.X[-1]`.
      # Process it in reverse order until an element exceeds the
      #   new value `x`. Then all earlier elements do too.

      u = self.vis_f[-1]
      
      if self.X[u] <= self.X[v]:
        # The new value `x` at vertex `v` blocks the value at vertex `u`. 
        if v == u+1:
          self._E += [[u, v, self.neighbour_weight]]
        else:
          self._E += [[u, v, self.X[u]-h]]
        del self.vis_f[-1]  # blocked vertices can never be seen again
        if self.X[u] == self.X[v]:
          break
        h = self.X[u]  # next weight will be height above the newly blocked `u`

      else:
        # The value `x` at vertex `v` adds one more edge to the existing HVG.
        if v == u+1:
          self._E += [[u, v, self.neighbour_weight]]
        else:
          self._E += [[u, v, self.X[v]-h]]
        break

    self.vis_f += [v]  # The new vertex `v` extends the decreasing subsequence.

    return self


  def add_batch(self, batch):
    '''
    Extend this HVG according to a new batch (list) of sequence values.
    '''

    for x in batch:
      self.add_one(x)

    return self


  @property
  def A(self):
    '''
    Sparse upper triangular weighted adjacency matrix of this HVG.

    Recomputed whenever new sequence values have been added.
    '''

    if (self._A is None) or (self._A.get_shape()[0] != len(self.X)):
      N = len(self.X)
      A = dok_matrix((N, N), dtype=np.float32)
      for u, v, w in self._E:
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

    # When using weights in merges the neighbour weights must match,
    # otherwise downstream weight-based graph analysis of the HVG is invalid.
    assert self.neighbour_weight == other.neighbour_weight

    # Set up the adjacency matrix for the joint graph
    L1, L2 = len(self), len(other)
    N = L1 + L2
    A = dok_matrix((N, N), dtype=np.float32)

    # We already have the upper left and lower right blocks.
    # These are already upper triangular.
    A[:L1, :L1] = self.A
    A[L1:, L1:] = other.A

    # But some nodes from the two graphs may be visible to one another.
    # We treat the possibly visible values as a subsequence and create its HVG.    
    keys = self.vis_f + [v + L1 for v in other.vis_p]  # relevant vertex labels
    vals = [self.X[v] for v in self.vis_f] + [other.X[v] for v in other.vis_p]
    hvg = HVG().add_batch(vals)

    # Edges of `hvg` give the off diagonal blocks in the joint adjacency matrix.
    for u, v, w in hvg._E:
      # We know u is in the current HVG and v is in the `other` HVG.
      # Moreover u and v are indirectly indexed via the list `keys`.
      A[keys[u], keys[v]] = w

    # Compute the combined time series and the new past/future node indices
    X = self.X + other.X
    vis_p = self.vis_p + [v + L1 for v in other.vis_p if v > self.max_val]
    vis_f = [v for v in self.vis_f if v > other.max_val]
    vis_f += [v + L1 for v in other.vis_f]
    max_val = max(self.max_val, other.max_val)

    if copy:
      hvg = HVG()
    else:
      hvg = self

    # Finally set all the properties on the larger HVG.
    hvg.X = X
    hvg.vis_p = vis_p
    hvg.vis_f = vis_f
    hvg.max_val = max_val
    hvg.neighbour_weight = self.neighbour_weight
    hvg._E = []  # Prefer adjacency representation when merges are involved.
    hvg._A = A

    return hvg



def hvg(X):
  return HVG(X)


