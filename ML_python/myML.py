"""
Functions for ML algorithms
""" 
import numpy as np
#  Calculate a cost function J
def computeCost(X_mxn, y, theta):
    m = y.shape[0]
    h_theta = X_mxn@theta
    error = h_theta-y
    J = 1/(2*m)*np.sum(error**2)    
    return J

def featNorm(X):
    X_norm = X;
    mu = np.zeros( (1, X.shape[1]) )
    sigma = np.zeros( (1, X.shape[1]) )
    
    mu = np.mean(X, axis=0)
    sigma = np.std(X, ddof=1, axis=0)
    X_norm = (X-mu)/sigma
    
    return X_norm, mu, sigma