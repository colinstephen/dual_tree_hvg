#!/usr/bin/env python
# -*- coding: UTF-8 -*-



"""
ORIGINAL ATTRIBUTION (this is an edited version)

AUTHOR: Delia Fano Yela
DATE:  December 2018
CONTACT: d.fanoyela@qmul.ac.uk
GIT: https://github.com/delialia/bst

REFERENCE FOR TECHNIQUE

"Fast transformation from time series to visibility graphs"
Xin Lan, Hongming Mo, Shiyu Chen, Qi Liu, and Yong Deng
Chaos 25, 083105 (2015); doi: 10.1063/1.4927835

DESCRIPTION

Provides a function to compute the horizontal visibility graph (HVG) of a time
series using a recursive divide and conquer approach.
"""



import numpy as np

def hvg(X, left=0, right=None, all_visible = None):

    if right is None:
        right = len(X)

    if all_visible == None : all_visible = []

    node_visible = []

    if left < right : # there must be at least two nodes in the time series
        # k = X[left:right].index(max(X[left:right])) + left
        k = np.argmax(X[left:right]) + left
        # check if k can see each node of series[left...right]

        for i in range(left,right):
            if i != k :
                a = min(i,k)
                b = max(i,k)

                yc = X[a+1:b]

                if all( yc[k] < min(X[a],X[b]) for k in range(b-a-1) ):
                    node_visible.append(i)
                elif all( yc[k] >= max(X[a],X[b]) for k in range(b-a-1) ):
                    break

        if len(node_visible) > 0 : all_visible.append([k, node_visible])

        hvg(X, left, k, all_visible = all_visible)
        hvg(X, k+1, right, all_visible = all_visible)

    return all_visible
