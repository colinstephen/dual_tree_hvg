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



class HVG:
  def __init__(self):
    self.X = []
    self.E = []
    self.vis_p = []
    self.vis_f = []
    self.max_val = -np.inf
    self.finite_lower_bound = -1e6

  def __len__(self):
    return len(self.X)

  def add_one(self, x):

    v = len(self.X)
    self.X += [x]
    # we now have self.X[v] = x
        
    if x > self.max_val:
      self.vis_p += [[v, self.max_val]]
      self.max_val = x

    while self.vis_f:
      
      u, bottom = self.vis_f[-1]
      bottom = bottom or self.finite_lower_bound
      
      if self.X[u] <= self.X[v]:
        self.E += [[u, v, self.X[u]-bottom]]
        del self.vis_f[-1]
        if self.X[u] == self.X[v]:
          break

      else:
        self.E += [[u, v, self.X[v]-bottom]]
        self.vis_f[-1][1] = self.X[v]
        break

    self.vis_f += [[v, None]]

  def add_batch(self, batch):
    for x in batch:
      self.add_one(x)

  def merge(self, other):
    self.add_batch(other.X)

  def hvg(self):
    # return a generator for the graph
    return ((u, v) for (u,v,w) in self.E)


def hvg(X, include_weights=True):
  '''
  Compute the HVG of a time series X in O(n) time.

  In practice we offload the work to a more general function that keeps track
  of additional structure. The additional structure is then forgotten.
  '''

  (V, E), (vis_p, vis_f), X = batch_hvg(X)

  if not include_weights:
    E = [(u, v) for (u, v, w) in E]

  return (V, E)



def batch_hvg(X):
  """
  Compute the HVG of a time series X in O(n) time.

  In addition, keep track of 'past' and 'future' visible points and any
  translations made to y-axis values.
  
  These data can be used to combine or merge batch HVGs together later.
  """

  # Adjust the y axis so the minimum value is one.
  translate = -np.min(X) + 1
  X = X + translate

  # Initialise the HVG graph
  V = xrange(len(X))  # vertices
  E = []              # edges
  vis_p = []          # vertices in V that can 'see the past' and their max
                      # preceding value
  vis_f = []          # vertices in V that can 'see the future' and their max
                      # succeeding value
  max_val = -np.inf   # the maximum value processed in the sequence X so far

  # Add vertices sequentially to the HVG by connecting edges back to earlier
  # visible vertices 
  for v in V:

    # Check whether we have a new global max
    if X[v] > max_val:
      # If so then the vertex v is past-facing above height max_val
      vis_p.append((v, max_val))
      max_val = X[v]

    # Check whether/which earlier vertices u can be seen from our new vertex v
    while len(vis_f) > 0:
      # keep track of the vertex index u and the lowest height any future vertex
      # needs to be to see u
      u, bottom = vis_f[-1]  

      # Check whether u is the rightmost vertex of the HVG so far
      if bottom == -np.inf:
        # If it is then to get consistent weights we use a basline
        bottom = 0

      # Check whether value at v is greater than or equal to value at u
      if X[u] <= X[v]:
        # If so then v can see u and u's 'shadow' on v is of height X[u]-bottom
        E.append((u, v, X[u]-bottom))
        # Moreover u is now blocked from any further future visibility
        vis_f = vis_f[:-1]

        # If the value at v was the same as at u then
        # no additional edges can be added from v to earlier vertices
        if X[u] == X[v]:
          break

      else:
        # The value at u was greater than the value at v. So v sees u
        # from height 'bottom' up to v's own height
        E.append((u, v, X[v]-bottom))
        
        # v partially blocks u so u's new 'bottom' is the height of v itself
        vis_f[-1] = (u, X[v])
        # moreover no further edges can be added
        break

    # the most recent vertex v is always future-facing
    vis_f.append((v, -np.inf))

  # Return the graph augmented with data on which vertices are past-facing or
  # future-facing. Also record by how much the original data were
  # shifted from their values during the computation.
  return (V, E), (vis_p, vis_f), X, translate



def merge(hvg_1, hvg_2):
  """
  Stitch together two batch HVGs using their future- and past-facing data.
  """

  (V1, E1), (vis_p1, vis_f1), X1, t1 = hvg_1
  (V2, E2), (vis_p2, vis_f2), X2, t2 = hvg_2

  # First ensure the vertex labels are disjoint    
  L1 = len(V1)
  L2 = len(V2)
  V = xrange(L1 + L2)
  E2 = [(u + L1, v + L1, w) for (u,v,w) in E2]
  vis_p2 = [(v + L1, val) for (v, val) in vis_p2]
  vis_f2 = [(v + L1, val) for (v, val) in vis_f2]

  # Account for possibly different translations to the original data.
  # Edge weights are relative so we only update Xi and vis_pi and vis_fi.
  if t1 < t2:
    # increase in X2 values was greater so add t2 to original X1 values
    align = t2 - t1
    X1 = X1 + align
    vis_p1 = [(v, val + align) for (v, val) in vis_p1]
    vis_f1 = [(v, val + align) for (v, val) in vis_f1]
    translate = t2

  elif t2 < t1:
    # increase in X1 values was greater so add t1 to original X2 values
    align = t1 - t2
    X2 = X2 + align
    vis_p2 = [(v, val + align) for (v, val) in vis_p2]
    vis_f2 = [(v, val + align) for (v, val) in vis_f2]
    translate = t1

  else:
    # translations t1 and t2 were the same
    translate = t1

  # Start with the past- and future-facing vertices from the first HVG
  vis_p = vis_p1

  # Find the maximum value in X1 to update past-facing visibility
  max_val = np.max(X1)

  # Initialise the _new_ edges that will join the HVGs together
  E_merge = []
  
  # Now process the past-facing vertices from the second batch.
  # Link them to the appropriate future-facing vertices in the first batch.
  for (v, val) in vis_p2:

    if X2[v-L1] > max_val:
      # update vis_p to include new past-facing vertices from hvg_2
      vis_p.append((v, max_val))
      max_val = X2[v-L1]

    while len(vis_f1) > 0:
      u, bottom = vis_f1[-1]

      # Check whether u is the rightmost vertex of the HVG so far
      if bottom == -np.inf:
        # If it is then to get consistent weights we use a basline
        bottom = 0

      # Check whether value at v is greater than or equal to value at u
      if X1[u] <= X2[v-L1]:
        # If so then v can see u and u's 'shadow' on v is of height X1[u]-bottom
        E_merge.append((u, v, X1[u]-bottom))
        # Moreover u is now blocked from any further future visibility
        vis_f1 = vis_f1[:-1]

        # If the value at v was the same as at u then
        # no additional edges can be added from v to earlier vertices
        if X1[u] == X2[v-L1]:
          break

      else:
        # The value at u was greater than the value at v. So v sees u
        # from height 'bottom' up to v's own height
        E_merge.append((u, v, X2[v-L1]-bottom))
        
        # v partially blocks u so u's new 'bottom' is the height of v itself
        vis_f1[-1] = (u, X2[v-L1])
        # moreover no further edges can be added
        break

    # Note we do note append anything to vis_f1 at this point as it
    # cannot increase in size during a merge.

  # However if there are remaining points in vis_f1 they must be future-facing
  # in the merged HVG over the maximum value in X2
  if len(vis_f1) > 0:
    (u, val) = vis_f1[-1]
    vis_f1[-1] = (u, np.max(X2))

  # Join the original edge sets with the 'merging edges'   
  E = E1 + E_merge + E2
  # Also update vis_f to include vertices in hvg_1 able to see the future 
  vis_f = vis_f1 + vis_f2
  # And ensure we pass back an aligned sequence for the joint batch
  X = np.concatenate((X1, X2))

  return (V, E), (vis_p, vis_f), X, translate


