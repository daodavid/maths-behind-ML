#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:59:48 2020

@author: daodeiv
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


rg = LogisticRegression(C=30)


class BinaryLogisticRegression():
    def __init__(self,lerning_rate=0.1,max_iter=1000):
        self.__learning_rate=lerning_rate
        self.__max_iter=1000
        self.__coef=None
        self.__intercept=None
        self.__initial_coefient=10
        
    def sigmoid(X, theta, intercept = 0 ):
        """
        takes:
        target vector X.X can be the matrix of many vectors and numer as well
        theta is an estimator vector (predict vector)
        intercept is theta zero elemt
        
        return result handled by sigmoid function, value can be array or number
    
         """
       #convertions to ndarray
        x = np.array(X)
        theta = np.array(theta)
        if len(x.shape) == 1:
            z = x*theta + intercept
        else:    
            z = x.dot(theta.T)+intercept # scallar product : <X|theta^(-1)> + intercept
            
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
    
    def fit(self, X, y):
        
        """ 
        fit the model according to given data dataset 
        """
        pass
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        """
        pass
    
    def predict(self,X):
        """
        Predict class labels for samples in X.
        """
        pass
    
    def get_coef(self):
        return self.coef
    
    def get_intercept(self):
        return self.intercept
    
    def set_params(self,args):
        pass
    