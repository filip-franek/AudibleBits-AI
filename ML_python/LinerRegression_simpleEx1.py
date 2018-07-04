# Example of Univariate Linear Regression
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:11:57 2018
@author: phill.oye
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Data preparation
# m ... number of training examples
# n ... number of features
text = open('ex1data1.txt','r')
#data_txt = data.readlines()
data = []
data_arr = np.zeros(())
for line in text:
    # take each line, remove whitespaces, and split elements separated with ','
    data.append([float(x) for x in line.strip().split(',')])

text.close()

m = len(data)
n = len(data[0])

#take last column of data as y vector and all previous as X matrix (x_1,x_2,...)
data_arr = np.array(data)
X = data_arr[:,n-2].reshape(m,n-1)
y = data_arr[:,n-1].reshape(m,1)
#assert(X.shape == (m,n-1))
#assert(y.shape == (m,1))

plt.figure(0)
plt.plot(X,y,'b*')
plt.xlabel('x_1')
plt.xlabel('y')
plt.title('Data')
plt.show()

X_avr = X.mean()
y_avr = y.mean()
#m = X.shape[0]
#n = X.shape[1]+1

# add column x_0 to X
X_mxn = np.insert(X,0,np.ones(X.size),axis=1)

# initialize values of theta parameters 
theta = np.zeros((n,1))

#%%  Train parameters
# gradient descend
iter = 1000
alpha = 0.01

# compute cost
h_theta = X_mxn@theta # hypothesis function

J = 1/(2*m)*np.sum((h_theta-y)**2) # cost function
#%% Gradient descend
J_hist = np.zeros((iter+1,1))
J_hist[0,0] = J
theta_hist = np.empty((iter+1,n))

for i in np.arange(0,iter):
    h = X_mxn@theta
    error = h-y
    theta_change = alpha/m*(X_mxn.T@error)
    theta = theta - theta_change
    h = X_mxn@theta
    error = h-y
    J_hist[i+1,0] = 1/(2*m)*np.sum((error)**2)
    
    
plt.show()
plt.plot(J_hist)
