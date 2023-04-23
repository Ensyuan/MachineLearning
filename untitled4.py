# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 20:56:41 2023

@author: Ensyuan
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy import stats


def gen_data(N=60):
    
    '''
 Generate simulated dataset of given number of data points
 
 Parameter
 ---------
 N : int
 number of data points
 
 Returns
 -------
 X : (N, 3) array
 Input data
 Y : (N, 3) array
 Output data
     '''
     
     ## design matrix
     X = np.zero((N, 3))
     X[:N//3, 0] = 1
     X[N//3:2*N//3, 1] = 1
     X[2*N//3:, 2] = 1
     
     
     ## simulated data for measurement results
     Y = np.random.normal(size=(N, 3))
     Y[N//3:2*N//3, 0] -= 0.5
     Y[2*N//3:, 0] += 0.5
     Y[N//3:2*N//3, 2] += 1
     Y[2*N//3:, 2] += 2
     
     return X, Y
 


#---------------------------------------
#  Contrast matrix for null hypothesis
#---------------------------------------

L = np.array([
    [1, -1, 0],
    [0, 1, -1]
    ])

#---------------------------------------
#  mGLM hypothesis testing
#---------------------------------------

def mglm_hyp_test_p(X,Y,L):
    '''
 Get p values of constraint in mGLM using 4 tests on dataset X,Y
 Parameters
 ----------
 X : (N, p) array >> Input data
 Y : (N, q) array >> Output data
 
 L : (k, p) array contrastive matrix for the constraint
 '''

    global mGLMs
    ## estimate parameters from data
    B = inv(X.T@X)@X.T@Y
    
    ## Additional covariance from assuming null hypothesis
    Qhyp = (L@B).T@inv(L@inv(X.T@X)@L.T)@(L@B)
    Qfull = (Y-X@B).T@(Y-X@B)
    
    ## various degree of freedom
    N, p = X.shape # input shape
    q = Y.shape[1] # output dimension
    ph = L.shape[0] # rank of contrast matrix
    s = min(q, ph)
    dfh = q*ph
    
    mGLMs = []  # collect all results

    # Pillai's trace 
    V = np.trace(inv(Qfull+Qhyp)@Qhyp)
    dfe = s*(N-p+s-q)
    mGLMs.append(["Pillai's trace","V", V, V/s, dfe])
    
    # Hotelling's trace
    T = np.trace(inv(Qfull)@Qhyp)
    dfe = s*(N-ph-2-q)+2
    mGLMs.append(["Hotelling's trace", "T", T, T/(T+s), dfe])
    
    
    # Wilk's Lambda
    













