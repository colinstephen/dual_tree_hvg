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


import sys
import numpy as np
from collections import ChainMap
from random import random



class HVG:
  '''
  Class for the weighted horizontal visibility graph (HVG) of a sequence.

  METHODS
  - add_one() : build HVG incrementally with new values 
  - add_batch() : call add_one() multiple times on a list of values
  - merge() : combine this HVG with another one
  
  OPERATORS
  - + and += : apply the merge method to existing HVGs
  
  PROPERTIES
  - A : Weighted upper triangular adjacency matrix
  '''


  def __init__(self, X_init=None):
    '''
    Sets up internal machinery for building HVGs from sequences.

    Optionally takes an initial sequence of values `X_init` to build the graph.
    '''

    self.V = {}  # stores vertices and their time series values {v: x}
    self.vis_p = []  # longest strictly increasing subsequence from `self.V[0]`
    self.vis_f = []  # longest strictly decreasing subsequence to `self.V[-1]`
    self.max_val = -np.inf  # update to avoid multiple calls to `max(self.V)`
    self.neighbour_weight = np.inf  # value of `w` for edges `[u, u+1, w]`
    self.E = {}  # stores weighted HVG edges `{(u, v): w}` for u,v in self.V

    # If a sequence was supplied, use it to build the graph.
    if X_init is not None and len(X_init):
      self.add_batch(X_init)


  def __iadd__(self, other):
    '''
    Allow extending HVGs via expressions like `hvg1 += hvg2`.
    '''

    return self.merge(other, copy=False)


  def __add__(self, other):
    '''
    Allow extending HVGs via expressions like `hvg3 = hvg1 + hvg2`.
    '''

    return self.merge(other, copy=True)


  def add_one(self, vx, v=None):
    '''
    Extend this HVG by adding edges induced by a new value `vx`.

    Optionally takes a vertex label for the new vertex.
    '''

    v = v or random()

    self.V[v] = vx
        
    if vx > self.max_val:
      # Update the longest strictly increasing subsequence from `self.V[0]`.
      self.vis_p += [(vx, v)]
      self.max_val = vx

    neighbour_added = False
    while self.vis_f:
      # There is a longest strictly decreasing subsequence to `self.V[-1]`.
      # Process it in reverse order until an element exceeds the
      #   new value `vx`. Then all earlier elements do too, so break out.

      ux, u = self.vis_f[-1]
      
      if ux <= vx:
        # The new value `vx` at vertex `v` blocks the value `ux` at vertex `u`. 
        if neighbour_added:
          self.E[u, v] = ux-h
        else:
          self.E[u, v] = self.neighbour_weight
          neighbour_added = True
        del self.vis_f[-1]  # blocked vertices can never be seen again
        if ux == vx:
          break
        h = ux  # next weight will be height above the newly blocked `u`

      else:
        # The value `vx` at vertex `v` adds one more edge to the existing HVG.
        if neighbour_added:
          self.E[u, v] = vx-h
        else:
          self.E[u, v] = self.neighbour_weight
        break

    self.vis_f += [(vx,v)]  # Vertex `v` extends the decreasing subsequence.

    return self


  def add_batch(self, batch, vertices=None):
    '''
    Extend this HVG according to a new batch (list) of sequence values.

    Optionally takes a list of vertex labels to apply to the new vertices. 
    '''
    if vertices is None:
      for x in batch:
        self.add_one(x)
    else:
      for x, v in zip(batch, vertices):
        self.add_one(x, v)

    return self


  def merge(self, other, copy=True):
    '''
    Merge an `other` HVG to the right of the current HVG.

    Optionally re-use and extend the current HVG instance with `copy=False`.
    '''

    assert self.neighbour_weight == other.neighbour_weight

    if copy:
      # Create a new HVG instance to return
      hvg = HVG()
      hvg.E = self.E
      hvg.V = self.V
    else:
      # We will extend the current HVG instance
      hvg = self

    # Use a ChainMap to store merged edge and vertex collections.
    if isinstance(hvg.E, dict):
      hvg.E = ChainMap(self.E)
    if isinstance(hvg.V, dict):
      hvg.V = ChainMap(self.V)

    # Concatenate the dicts of edges and the time series
    hvg.E = hvg.E.new_child(other.E)
    hvg.V = hvg.V.new_child(other.V)

    # Some nodes from the two input graphs will be visible to one another.
    # We treat the possibly-visible values as a subsequence and create its HVG.    
    join_vtx = self.vis_f + other.vis_p  # list of (vx,v) value-vertex pairs
    join_hvg = HVG()
    vals, vertices = zip(*join_vtx)
    join_hvg.add_batch(vals, vertices)

    # Now use this joining HVG to infer the extra edges in the merged HVG. 
    merge_edges = {}
    merge_edges[self.vis_f[-1][1], other.vis_p[0][1]] = self.neighbour_weight
    for (u, v), w in join_hvg.E.items():
      if w != self.neighbour_weight:
        merge_edges[u, v] = w
    hvg.E = hvg.E.new_child(merge_edges)
    
    # Compute the rest of the properties on the returned instance.
    vis_p = []
    for vx, v in reversed(other.vis_p):
      if vx > self.max_val:
        vis_p += [(vx, v)]
      else:
        break
    hvg.vis_p = self.vis_p + list(reversed(vis_p))

    vis_f = []
    for vx, v in self.vis_f:
      if vx > other.max_val:
        vis_f += [(vx, v)]
      else:
        break
    hvg.vis_f = vis_f + other.vis_f

    hvg.max_val = max(self.max_val, other.max_val)
    hvg.neighbour_weight = self.neighbour_weight

    return hvg


