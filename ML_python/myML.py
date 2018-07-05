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