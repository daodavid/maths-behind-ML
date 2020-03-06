# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:17:52 2020

@author: Daodavid
"""
import numpy as np
import matplotlib.pyplot as plt

class DaoRegression():
    
    
    
    def __init__(self,lerning_rate=0.1,max_iter=1000):
        self.l_rate=lerning_rate
        self.max_iter=1000
        self.coef=None
        self.intercept=None
        self.initial_coefient=10
        
        
    def fit(self, X, y):
        
        """ 
        fit the model according to given data dataset and prediction function
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
    
    
    
    

class DaoBinaryRegression(DaoRegression):
    pass

    def fit(self,X,y):
        self.t_data=X
        self.label=y
        self.t_data=np.array(self.t_data).T
        self.coef=np.full(a.shape[1], self.initial_coefient)
        self.intercept=self.initial_coefient
        
        
        
    def sigmoid(coef,intecept,x):
        z = coef*x + intecept
        return 1/(1.0+np.exp(-z))
    
    
#print(np.array([[1,2,3],[2,3,4]]).shape)    
a = np.array([[1,2,3],[2,3,4]])
b = a.dot(np.array([1,2,3]))
print(b[1])