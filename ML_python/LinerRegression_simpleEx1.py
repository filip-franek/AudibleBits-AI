# Example of Univariate Linear Regression
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:11:57 2018
@author: phill.oye
"""

import numpy as np
import matplotlib.pyplot as plt
import myML

plt.show()
#%% Data preparation
# m ... number of training examples
# n ... number of features
text = open('ex1data1.txt','r')
#data_txt = data.readlines()i
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
plt.plot(X,y,'b*') # label='Training data'
plt.xlabel('x_1')
plt.ylabel('y')
plt.title('Data')

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
error = h_theta-y
J = 1/(2*m)*np.sum(error**2) # cost function
#%% Gradient descend
J_hist = np.empty((iter+1,1))
J_hist[0,0] = J
theta_hist = np.zeros((n,iter+1))
#theta_hist[:,0] = theta

for i in np.arange(0,iter):
    h_theta = X_mxn@theta
    error = h_theta-y
    theta_change = alpha/m*(X_mxn.T@error)
    theta = theta - theta_change
    h_theta = X_mxn@theta
    error = h_theta-y
    J_hist[i+1,0] = 1/(2*m)*np.sum(error**2)
    theta_hist[:,i+1] = theta.reshape(n,)
    # theta_hist[:.i+1] becomes rank 0 with size (n,), thus reshaping

#%% plot linear regression, predicted points, cost and theta contours
plt.figure(0)
plt.plot(X,h_theta,'r-') # label='Linear regression'

# Predict data points
x_p1 = X.min()+(X.max()-X.min())/2
x_p2 = X.min()+(X.max()-X.min())/100*90
x_p3 = X.min()+(X.max()-X.min())/100*10
X_p = np.array([x_p1, x_p2, x_p3]).reshape(3,1)
X_o = np.ones((3,1))
X_op = np.append(X_o, X_p, axis=1)
y_p = X_op @ theta

plt.plot(X_p, y_p, 'ks') # label='Predicted values'
plt.legend(['Training data', 'Linear regression', 'Predicted values'])

plt.figure(1)
plt.plot(J_hist)
plt.title('Cost for ' + str(iter) + ' iterations and alpha = ' + str(alpha))
plt.xlabel('# iterations')
plt.ylabel('Cost J')
#%% Plot 3D thetas
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-4, 4, 100)
#Th1, Th2 = np.meshgrid(theta0_vals,theta1_vals)
J_vals = np.zeros((100,100))

for i in range(theta0_vals.shape[0]):
    for j in range(theta1_vals.shape[0]):
        theta_vals = np.array([theta0_vals[i], theta1_vals[j]]).reshape(n,1)
        J_vals[i,j] = myML.computeCost(X_mxn, y, theta_vals)

plt.figure(3)
plt.plot(theta_hist[0,:],theta_hist[1,:],'r-x')
cp = plt.contour(theta0_vals, theta1_vals, J_vals.T, np.logspace(-1, 2, 20))
#plt.clabel(cp, inline=True, fontsize=10)
plt.title('Contour plot of cost with respect to thetas')
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.legend(['Gradient descent'])

plt.figure(4)
np.meshgrid