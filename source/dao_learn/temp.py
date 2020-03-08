# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def sigmoid(X, theta, intercept = 0 ):
    """
    Target vector X,X can be the matrix of many vectors and numer as well
    theta is an estimator vector
    intercept is theta zero elemt
    
    """
    #convertions to ndarray
    X = np.array(X)
    theta = np.array(theta)
    if len(x.shape) == 1:
        z = x*theta + intercept
    else:    
        z = X.dot(theta.T)+intercept # scallar product : <X|theta^(-1)> + intercept
    return 1/(1+np.exp(-z)) #sigmoid transformation of z
    

def lost(arg, y_target, x_i=1):
    """
    takes arg ,that is the result of sigmoid it has to be array
    y_target label variable which is 1 or 0
    x_i lement is every i element from X vectors in one array [x[i][j]] j is constant refer to column related to j_estimator
    """
    y = np.array(y_target)
    x_i = np.array(x_i)
    return (arg-y)*x_i


def cost(X, estimators, Y_label, intecept, x_i=1):
    """
    takes:
    X is Target vectors 
    estimators are our fitin parametes theta_i
    Y_label is our target values zero or one
    x_i is the i_th element (column) of target element related to Theta i_th estimator
    """
    m = Y_label.shape[0]
    n = np.array(x_i).shape[0]
    if m != n :
        print(n)
        raise ValueError('x_i and Y_label must have same shape')
    sigmoid_result = sigmoid(X, estimators, intecept)    
    result = lost(sigmoid_result, Y_label, x_i)
    return result.sum()   
  

def gradient_descent(X_data, Y_label, times_interaction=900, learning_rate=0.01, init_value=10):
    x = np.array(X_data)
    y_l = np.array(Y_label)
    s =x.shape
    len_s = len(s)
    m = x.shape[0]
    if len_s>1:
        n = x.shape[1]
    else :
        n=1
    
    intercept = init_value
    esimators = np.full(n, init_value)
    a_args = []
    b_args = []
    for i in range(times_interaction):
        for j in range(len(esimators)):
            x_column = x[:,j] if len_s>1 else  x
            esimators[j]-=cost(x, esimators, y_l, intercept, x_column )*learning_rate
            a_args=np.append(a_args, esimators[j])
        
            
        intercept -=cost(x, esimators, y_l, intercept, np.full(m, 1) )*learning_rate     
        b_args=np.append(b_args, intercept)
        
        
    print(a_args)
    print(b_args)
    print("coef:",esimators)
    print("intercept:",intercept)
              




#define z argument as lininear euation of z = 2*x + 4
z_f = lambda x : 10*x+4
x = np.linspace(-10,10,30)

z_args = np.array([z_f(i) for i in x])


y_prime = np.array( 1/(1+np.exp(-z_args)))


## generate labeled  from already define y_prime data given the sigmoid with args z = 2*x+4
def generated_label(i):
        if i < 0.5 :
            return 0
        elif i > 0.5 :
            return 1
        else :
            return np.random.randint(0,2)
        
 ####       

#####

#label = data[:,1] #.reshape(-1, 1)
#trained_data = data[:,0]
y_label = np.array([generated_label(i) for i in y_prime])


print("Y______original",y_label)
gradient_descent(x,y_label,times_interaction=6000,learning_rate=0.01)        
rg = LogisticRegression(C=30)

rg.fit(x.reshape(-1, 1),y_label)
print(rg.coef_)
print(rg.intercept_)



z_f = lambda x : 10*x+6
x = np.linspace(-10,10,30)

z_args = np.array([z_f(i) for i in x])


y_prime = np.array( 1/(1+np.exp(-z_args)))

y_label = np.array([generated_label(i) for i in y_prime])
print("Y______predict",y_label)