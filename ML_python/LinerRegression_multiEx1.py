# Example of Univariate Linear Regression
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:11:57 2018
@author: phill.oye
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import myML

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline') # for inline plotting
#get_ipython().run_line_magic('matplotlib', 'qt') # for plotting in external window

plt.close('all')
plt.show()
#%% Data preparation
# m ... number of training examples
# n ... number of features
text = open('ex1data2.txt','r')
#data_txt = data.readlines()i
data = []
for line in text:
    # take each line, remove whitespaces, and split elements separated with ','
    data.append([float(x) for x in line.strip().split(',')])

text.close()

m = len(data)
n = len(data[0]) #

#take last column of data as y vector and all previous as X matrix (x_1,x_2,...)
data_arr = np.array(data)
X = data_arr[:,0:n-1].reshape(m,n-1) # assert(X.shape == (m,n-1))
y = data_arr[:,n-1].reshape(m,1)

fig1, ax1 = plt.subplots(nrows=n-1, ncols =1) # squeeze=False
for i in range(n-1):
    ax1[i] = (plt.subplot(n-1,1,i+1))
    lines1, = ax1[i].plot(X[:,i],y,'*') # label='Training data'
    ax1[i].set_xlabel('x_'+str(i+1))
    ax1[i].set_ylabel('y')
#    plt.legend(['x_'+str(i+1)])
    ax1[i].set_title('Data x_'+str(i+1))
plt.tight_layout()
#%% normalization
if n > 2:
    X_norm, mu, sigma = myML.featNorm(X)

# add column x_0 to X
X_mxn = np.insert(X_norm,0,np.ones(X.shape[0]),axis=1)

# initialize values of theta parameters 
theta = np.zeros((n,1))
#%%  Train parameters
# gradient descend
iter = 400
alpha = 0.05

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

for i in range(n-1):
    lines00, = ax1[i].plot(X[:,i],h_theta,'r*')
# linear regression plat needs to be in 3d with 3 theta parameters
#%%
# Predict data points
x_p0 = np.array([1650,3])   # ft*2, floors
x_p1 = X.min(axis=0)+(X.max(axis=0)-X.min(axis=0))/2        # middle point
x_p2 = X.min(axis=0)+(X.max(axis=0)-X.min(axis=0))/100*90   # 90%
x_p3 = X.min(axis=0)+(X.max(axis=0)-X.min(axis=0))/100*10   # 10%
X_p = np.array([x_p0, x_p1, x_p2, x_p3]).reshape(-1,n-1)
# need to normalize the new set with saved mu and sigma values
X_p_norm = (X_p-mu)/sigma;
# x_0 does not need to be normalized
X_o = np.ones((X_p.shape[0],1))
X_op_norm = np.append(X_o, X_p_norm, axis=1)
y_p = X_op_norm @ theta

for i in range(n-1):
    ax1[i].plot(X_p[:,i],y_p,'ks')
    ax1[i].legend(['Training data', 'Linear regression', 'Predicted values'])

fig2, ax2 = plt.subplots(nrows=1,ncols=1)
ax2.plot(J_hist)
ax2.set_title('Cost for ' + str(iter) + ' iterations and alpha = ' + str(alpha))
ax2.set_xlabel('# iterations')
ax2.set_ylabel('Cost J')
#%% Plot 3D thetas
theta0_vals = np.linspace(-1e7, 1e7, 100)
theta1_vals = np.linspace(-1e7, 1e7, 100)
#theta2_vals = np.linspace(-1e7, 1e7, 100)
theta2_vals = np.array([-6639.78])

J_vals = np.zeros((100,100))

for i in range(theta0_vals.shape[0]):
    for j in range(theta1_vals.shape[0]):
        theta_vals = np.array([theta0_vals[i], theta1_vals[j], theta2_vals]).reshape(n,1)
        J_vals[i,j] = myML.computeCost(X_mxn, y, theta_vals)

fig3, ax3 = plt.subplots(nrows=1, ncols=1)
ax3.plot(theta_hist[0,:],theta_hist[1,:],'r-x')
ax3.contour(theta0_vals, theta1_vals, J_vals.T)

#plt.clabel(cp, inline=True, fontsize=10)
plt.title('Contour plot of cost with respect to thetas')
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.legend(['Gradient descent'])
#%%
fig = plt.figure(4)
ax = fig.gca(projection='3d')
Th0, Th1 = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(Th0, Th1, J_vals)
#%%
fig = plt.figure(5)
ax = fig.add_subplot(111, projection='3d')
# Plot a basic wireframe.
ax.plot_wireframe(Th0, Th1, J_vals, rstride=10, cstride=10)
#%% 
fig = plt.figure(6)
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(Th0, Th1, J_vals, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
