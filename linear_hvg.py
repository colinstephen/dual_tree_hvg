#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

def linear_hvg(X):
    _, E, _ = batch_hvg(X)
    return E

def batch_hvg(X, finite_lower_bound=0):
    """
    Returns the 'past' and 'future' visible points as well as the HVG graph edges.
    These can be used to combine or merge batch HVGs together.

    The finite_lower_bound assumes all values in X are bounded below by this value.
    It is only important when merging HVGs to keep the weight structure of the merged graph
    consistent with its components. It does not affect standard unweighted HVG merges.
    """

    E = []
    vis_p = []
    vis_f = []
    max_val = -np.inf

    for v in np.arange(len(X)):
        if X[v] > max_val:
            vis_p.append((v, max_val))  # the past can see v in range (max_val, X[v]]
            max_val = X[v]
        while len(vis_f) > 0:
            u, bottom = vis_f[-1]
            if bottom == -np.inf:
                bottom = finite_lower_bound
            if X[u] <= X[v]:  # value at v is greater than or equal to value at u
                E.append((u, v, X[u]-bottom))  # v can see u in range (bottom, top]
                vis_f = vis_f[:-1]  # u cannot see beyond v
                if X[u] == X[v]:  # no further edges can be added
                    break
            else:  # value at v is less than value at u
                E.append((u, v, X[v]-bottom))  # v can see u in range (bottom, X[v]]
                vis_f[-1] = (u, X[v])  # v blocks u in range (bottom, X[v]] so future can now see u in range (X[v], top]
                break  # no further edges can be added
        vis_f.append((v, -np.inf))  # the future can see all of v

    return vis_p, E, vis_f
