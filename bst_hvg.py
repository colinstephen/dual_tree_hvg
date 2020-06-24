#!/usr/bin/env python
# -*- coding: UTF-8 -*-



"""
ORIGINAL ATTRIBUTION (this is an edited version)

AUTHOR: Delia Fano Yela
DATE:  December 2018
CONTACT: d.fanoyela@qmul.ac.uk
GIT: https://github.com/delialia/bst 

REFERENCE FOR TECHNIQUE

Fano Yela, D., Thalmann, F., Nicosia, V., Stowell, D., Sandler, M., 2020.
"Online visibility graphs: Encoding visibility in a binary search tree."
Phys. Rev. Research 2, 023069. https://doi.org/10.1103/PhysRevResearch.2.023069

DESCRIPTION

Provides a class and functions to compute the horizontal visibility graphs
(HVG) of time series using a recursive binary search tree encoding-decoding
approach.
"""



import numpy as np



class Node:

  def __init__(self, value=None, data=None, left=None, right=None):
    self.value  = value     # Float: Value of the node
    self.data   = data      # Float: Data value associated with that node
    self.left   = left      # Node: Left child
    self.right  = right     # Node: Right child

  def __eq__(self, other):
    """Override the default Equals behavior"""
    if isinstance(other, self.__class__):
      return self.value == other.value and self.data == other.data
    else:
      return False

  def __ne__(self, other):
    """Override the default Unequal behavior"""
    return self.value != other.value or self.data != other.data

  def add(self, node):
    if  not self.value and self.value != 0:
      self.value = node.value
      self.data = node.data
    else:
      if node.value < self.value:
        if not self.left and self.left != 0 : self.left = Node()
        self.left.add(node)
      else:
        if not self.right and self.right != 0 : self.right = Node()
        self.right.add(node)

  def getKids(self):
    if not self.left and not self.right:
      return None
    kids = {}
    if self.left:
      kids["left"] =  self.left
    if self.right:
      kids["right"] = self.right
    return kids

  def delKid(self, keyword):
    if keyword == 'left': self.left = None
    if keyword == 'right': self.right = None

  def __add__(self, other):
    return merge([self, other])



def hvg(X, sort='mergesort'):
  '''
  use 'mergesort' for worst case O(n log n)
  or 'quicksort' for worst case O(n^2) but perhaps better average performance
  '''

  root = Node()

  sorted_indexes = np.argsort(X, kind=sort)[::-1]
  sorted_data = X[sorted_indexes]

  for (index, data) in zip(sorted_indexes, sorted_data):
    root.add(Node(value = index, data = data))

  return root



def sign(x):
    if x > 0:
        return 1.
    elif x < 0:
        return -1.
    elif x == 0:
        return 0.
    else:
        return x



def merge(list_roots_in): #[node01, node02]
  
  if len(list_roots_in) == 0:
    return None
  elif len(list_roots_in) == 1:
    return list_roots_in[0]
  else:
    # sort in ascending order by the node values in list:
    vroots = [x.value for x in list_roots_in]
    sidx = sorted(range(len(vroots)), reverse=False, key=lambda k: vroots[k])
    list_roots = [list_roots_in[i] for i in sidx]
    # find the node with the maximum data point in the list:
    # index --> Find index of first " " in mylist
    root = list_roots[[x.data for x in list_roots].index(max([x.data for x in
      list_roots]))]
    # Get the children of the maximum root
    root_kids = root.getKids()
    # Set the threshold :
    th =  root.value
    # initialise the pool of nodes to process:
    iter_pool = list_roots[:]
    iter_pool.remove(root)
    pool = iter_pool[:]
    # add the kids from the leftovers that are on the opposite side of the
    # threshold to their parent
    for node in iter_pool:
      kids = node.getKids()
      if kids:
        for k in kids.items():
          if sign(node.value - th) != sign(k[1].value - th):
            pool.append(k[1])
            node.delKid(k[0])

    # add the children of the maximum node
    if bool(root_kids):
      for k in root_kids.items():
        pool.append(k[1])
        root.delKid(k[0])

    # divide the pool according to the threshold
    smaller = [ x for x in pool if x.value < th]
    bigger  = [ x for x in pool if x.value > th]

    # return the node recursively:
    return(Node(value = root.value, data = root.data, left = merge(smaller),
      right = merge(bigger)))


