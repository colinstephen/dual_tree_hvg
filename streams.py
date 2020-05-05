#!/usr/bin/env python
# -*- coding: UTF-8 -*-



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Colin Stephen
# DATE:  April 2020
# CONTACT: colin.stephen@coventry.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""
Provides functions to create and consume streams of time series and convert them
in to horizontal visibility graphs (HVGs).
"""



import time
import numpy as np
from scipy.integrate import odeint
from fbm import FBM, MBM



def random(n, seed=None):
	np.random.seed(seed)
	return np.random.random(size=n)



def discrete_random_walk(n, seed=None):
    
    np.random.seed(seed)
    walk = np.empty(n)
    walk[0] = 0
    x = np.random.choice([-1,1], size=n-1)
    walk[1:] = np.cumsum(x)
    return walk



def fbm(n, hurst=0.5, seed=None):

    np.random.seed(seed)

    f = FBM(n, hurst)
    fbm_sample = f.fbm()
    return fbm_sample


def mfbm(n, hurst_func, seed=None):

    np.random.seed(seed)

    f = MBM(n, hurst=hurst_func)
    mfbm_sample = f.mbm()
    return mfbm_sample


def logistic_attractor(n, a=3.9995, delay=1000, seed=None):
    
    np.random.seed(seed)

    logistic = lambda x: a * x * (1-x)

    x_ = np.random.random()
    for i in range(delay):
        x_ = logistic(x_)

    x = np.empty(n)    
    x[0] = x_

    for i in range(n - 1):
            x[i+1] = logistic(x[i])
    
    return x



def lorentz_attractor(n, sigma=10, beta=8.0/3, rho=28, seed=None):
    # return the x component of a 3d Lorenz system
    
    # initial condition
    X = np.random.random(3)
    
    def lorenz(X, t, sigma, beta, rho):
        """The Lorenz equations."""
        u, v, w = X
        up = -sigma*(u - v)
        vp = rho*u - v - u*w
        wp = -beta*w + u*v
        return up, vp, wp

    tmax = 100
    t = np.linspace(0, tmax, n)
    f = odeint(lorenz, X, t, args=(sigma, beta, rho))
    x, y, z = f.T

    return x
    


def henon_attractor(n, alpha=1.4, beta=0.3035):
    
    # Henon map
    def henon(uv):
        u, v = uv
        up = 1 - alpha*u**2 + v
        vp = beta*u
        return up, vp

    X = np.empty((n,2))
    X[0] = np.random.random(2)

    for i in range(n - 1):
        X[i+1] = henon(X[i])

    return X[:,0]



def packet_stream(dynamic_model, *args, n_batches=1, batch_size=1, **kwargs):
    N = n_batches * batch_size
    data = stream(*args, **kwargs)
    for i in range(n_batches):
        yield data[i*batch_size : (i+1)*batch_size]


