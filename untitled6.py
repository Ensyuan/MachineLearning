# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 15:42:16 2023

@author: Ensyuan
"""

##  file:///D:/%E7%A5%9E%E7%B6%93%E7%A7%91%E5%AD%B8%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90/hw4sol.pdf
##  file:///D:/神經科學資料分析/hw4sol.pdf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from matplotlib import cm

#---------------------------
# Datasets
#---------------------------


N = 200
np.random.seed(123)

mu1 = np.array([[0,0],[-4,4],[4,5]])
si1 = np.array([[[1, 0.2], [0.2, 1]], [[1, 0.2], [0.2, 1]], [[1, 0.2], [0.2, 1]]])
X1 = [np.random.multivariate_normal(m, s, N) for m, s in zip(mu1, si1)]


mu2 = np.array([[0,0],[-3, 3],[3, 3]])
si2 = np.array([[[4, 1.2], [1.2, 1.5]], [[2, 0], [0, 3]], [[1, 1], [1, 4]]])
X2 = [np.random.multivariate_normal(m, s, N) for m, s in zip(mu2, si2)]

X3 = [np.array([x for x in 
                np.random.multivariate_normal([0,0],[[6,0],[0,3]], 5*N)
                if x[1]<x[0]**2]), 
      np.array([x for x in 
                np.random.multivariate_normal([0,0],[[6,0],[0,3]], 10*N) 
                if x[1]>x[0]**2])
    ]

def combine_data(X):
    x = np.concatenate(X)
    y = np.concatenate([[k]*len(X[k]) for k in range(len(X))])
    return x,y,len(X)



#---------------------------
# Clustering
#---------------------------

from scipy.cluster.vq import kmeans

def classify_with_kmeans_sp(X,K):
    mus = kmeans(X,K)[0]
    return np.array([
        np.argmin([np.sum((x-mu)**2) for mu in mus]) # np.argmin??
        for x in X
        ])

def classify_with_kmeans_cc(X,K):
    y = np.random.randint(K, size=len(X))
    for i in range(100):
        mu = [np.mean(X[y==k], axis=0) for k in range(K)]
        d = np.array([
            (lambda v:np.einsum('ij,ij->i', v, v))(X-mu[k]) # np.einsum??
            for k in range(K)
            ])
        new_y = np.argmin(d, axis=0)
        if np.all(y==new_y):break
        y = new_y
        
    print(f'Stop after {i} iterations.')
    return y

for XX in [X1, X2, X3]:
    X, y, K = combine_data(XX)
    yy = classify_with_kmeans_sp(X, K)
    f, [a, b] = plt.subplots(1, 2, figsize = (6.5, 3))
    a.scatter(*X.T, c=y, cmap='viridis', s = 2, alpha = 0.5)
    a.set_xlabel('$x_1$')
    a.set_ylabel('$x_2$')
    b.scatter(*X.T, c=yy, s = 2, alpha = 0.5)
    b.set_xlabel('$x_1$')
    plt.show()
    
    
X, y, K = combine_data(X3)
yy = classify_with_kmeans_sp(X,K)

f, [a, b] = plt.subplots(1, 2, figsize=(6.5, 3))
a.scatter(*X.T, c=y, cmap='viridis', s = 2, alpha = 0.5)
a.set_xlabel('$x_1$')
a.set_ylabel('$x_2$')
b.scatter(*X.T, c=yy, s = 2, alpha = 0.5)
b.set_xlabel('$x_1$')
plt.show()



X, y, K = combine_data(X3)
yy = classify_with_kmeans_cc(X,K)

f, [a, b] = plt.subplots(1, 2, figsize=(6.5, 3))
a.scatter(*X.T, c=y, cmap='viridis', s = 2, alpha = 0.5)
a.set_xlabel('$x_1$')
a.set_ylabel('$x_2$')
b.scatter(*X.T, c=yy, s = 2, alpha = 0.5)
b.set_xlabel('$x_1$')
plt.show()















