import numpy as np

def dc_hvg(X, left, right, all_visible = None):

    if all_visible == None : all_visible = []

    node_visible = []

    if left < right : # there must be at least two nodes in the time series
        # k = X[left:right].index(max(X[left:right])) + left
        k = np.argmax(X[left:right]) + left
        # check if k can see each node of series[left...right]

        for i in xrange(left,right):
            if i != k :
                a = min(i,k)
                b = max(i,k)

                yc = X[a+1:b]

                if all( yc[k] < min(X[a],X[b]) for k in xrange(b-a-1) ):
                    node_visible.append(i)
                elif all( yc[k] >= max(X[a],X[b]) for k in xrange(b-a-1) ):
                    break

        if len(node_visible) > 0 : all_visible.append([k, node_visible])

        dc_hvg(X, left, k, all_visible = all_visible)
        dc_hvg(X, k+1, right, all_visible = all_visible)

    return all_visible
