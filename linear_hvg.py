#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

def linear_hvg(X):
    (V, E), (vis_p, vis_f), X = batch_hvg(X)
    return E

def batch_hvg(X, finite_lower_bound=0):
    """
    Returns the 'past' and 'future' visible points as well as the HVG graph edges.
    These can be used to combine or merge batch HVGs together.

    The finite_lower_bound assumes all values in X are bounded below by this value.
    It is only important when merging HVGs to keep the weight structure of the merged graph
    consistent with its components. It does not affect standard unweighted HVG merges.
    """

    V = xrange(len(X))
    E = []
    vis_p = []
    vis_f = []
    max_val = -np.inf

    for v in xrange(len(X)):
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

    return (V, E), (vis_p, vis_f), X

def merge_batch_hvgs(hvg_1, hvg_2, finite_lower_bound=0):
    """
    Stitch together two HVGs using their future and past visible vertices.
    """
    (V1, E1), (vis_p1, vis_f1), X1 = hvg_1
    (V2, E2), (vis_p2, vis_f2), X2 = hvg_2

    # First ensure the vertex labels are disjoint    
    L1 = len(V1)
    L2 = len(V2)
    V = xrange(L1+L2)
    E2 = [(u+L1, v+L1, w) for (u,v,w) in E2]
    vis_p2 = [(v+L1, val) for (v, val) in vis_p2]
    vis_f2 = [(v+L1, val) for (v, val) in vis_f2]

    # Start with the past and future looking vertices from the first batch
    vis_p = vis_p1
    max_val = np.max(X1)  # TODO: could get this from vis_p1 or vis_f1

    # Now process the past looking vertices from the second batch.
    # Link them to the appropriate vertices in the first batch.
    E_merge = []  # This will contain the *new* edges merging the HVGs together
    for (v, val) in vis_p2:
        if X2[v-L1] > max_val:
            # update vis_p as necessary
            vis_p.append((v, max_val))
            max_val = X2[v-L1]
        while len(vis_f1) > 0:
            u, bottom = vis_f1[-1]
            if bottom == -np.inf:
                bottom = finite_lower_bound
            if X1[u] <= X2[v-L1]:  # value at v is greater than or equal to value at u
                E_merge.append((u, v, X1[u]-bottom))  # v can see u in range (bottom, top]
                vis_f1 = vis_f1[:-1]  # u cannot see beyond v
                if X1[u] == X2[v-L1]:  # no further edges can be added
                    break
            else:  # value at v is less than value at u
                E_merge.append((u, v, X2[v-L1]-bottom))  # v can see u in range (bottom, X[v]]
                vis_f1[-1] = (u, X2[v-L1])  # v blocks u in range (bottom, X[v]] so future can now see u in range (X[v], top]
                break  # no further edges can be added to v
        # Note we do note append anything to vis_f1 at this point as it is already fully computed
    # However if there are remaining points in vis_f1 they will 'see over' the points in X2
    if len(vis_f1) > 0:
        vis_f1[-1][1] = np.max(X2)  # TODO: could get this from vis_p2 or vis_f2
    
    E = E1 + E_merge + E2
    vis_f = vis_f1 + vis_f2
    X = np.concatenate((X1, X2))
    return (V, E), (vis_p, vis_f), X

