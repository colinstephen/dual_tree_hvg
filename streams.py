#!/usr/bin/env python
# -*- coding: UTF-8 -*-



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Colin Stephen
# DATE:  April 2020
# CONTACT: colin.stephen@coventry.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""
Provides functions to create streams of interesting time series.
"""



import time
import numpy as np
from scipy.integrate import odeint
from fbm import FBM, MBM



def random(n, seed=None):
    # random noise
	np.random.seed(seed)
	return np.random.random(size=n)



def discrete_random_walk(n, seed=None):
    # single step random walk
    np.random.seed(seed)
    walk = np.empty(n)
    walk[0] = 0
    x = np.random.choice([-1,1], size=n-1)
    walk[1:] = np.cumsum(x)
    return walk



def fbm(n, hurst=0.5, seed=None):
    # fractional Brownian motion
    np.random.seed(seed)
    f = FBM(n=n, hurst=hurst)
    fbm_sample = f.fbm()
    return fbm_sample



def mfbm(n, hurst_func, seed=None):
    # multifractional Brownian motion
    # hurst_func is a function mapping real time to real Hurst index
    np.random.seed(seed)
    f = MBM(n, hurst=hurst_func)
    mfbm_sample = f.mbm()
    return mfbm_sample



def logistic_attractor(n, a=3.9995, delay=1000, seed=None):
    # discrete logistic map
    logistic = lambda x: a * x * (1-x)

    # burn in the trajectory
    np.random.seed(seed)
    x_ = np.random.random()
    for i in range(delay):
        x_ = logistic(x_)

    # now generate the actual data
    x = np.empty(n)    
    x[0] = x_

    for i in range(n - 1):
        x[i+1] = logistic(x[i])
    
    return x



def standard_map(n, K=1.2, seed=None):
    X = np.empty((n,2))
    np.random.seed(seed)
    X[0] = np.random.random(2)

    def standard(xy):
        p, theta = xy
        p = np.mod(p, 2*np.pi)
        theta = np.mod(theta, 2*np.pi)
        p_ = p + K * np.sin(theta)
        theta_ = theta + p_
        return p_, theta_

    for i in range(n-1):
        X[i+1] = standard(X[i])

    return X[:,0]



def lorenz_attractor(n, sigma=10, beta=8.0/3, rho=28, seed=None, tmax=100):
    # samples from the x component of a 3d Lorenz system
    
    # random initial condition
    np.random.seed(seed)
    X = np.random.random(3)
    
    def lorenz(X, t, sigma, beta, rho):
        """The Lorenz equations."""
        u, v, w = X
        up = -sigma*(u - v)
        vp = rho*u - v - u*w
        wp = -beta*w + u*v
        return up, vp, wp

    # now solve the Lorenz equations to get the trajectory
    t = np.linspace(0, tmax, n)
    f = odeint(lorenz, X, t, args=(sigma, beta, rho))
    x = f[:,0]

    return x
    


def rossler_attractor(n, a=0.2, b=0.2, c=5.7, seed=None, tmax=100):
    # samples from the z component of a 3d Rossler system

    # random initial condition
    np.random.seed(seed)
    X = np.random.random(3)
    
    def rossler(X, t, a, b, c):
        """The Rossler equations."""
        x, y, z = X
        ux = -(y + z)
        uy = x + a * y
        uz = b + z * (x - c)
        return ux, uy, uz

    # now solve the Rossler equations to get the trajectory
    t = np.linspace(0, tmax, n)
    f = odeint(rossler, X, t, args=(a, b, c))
    z = f[:,2]

    return z



