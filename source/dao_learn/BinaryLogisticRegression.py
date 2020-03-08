#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:59:48 2020

@author: daodeiv
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


rg = LogisticRegression(C=30)


class BinaryLogisticRegression():
    def __init__(self,learning_rate=0.1,max_iter=1000):
        self.learning_rate=learning_rate
        self.__max_iter=1000
        self.coef=None
        self.intercept=10
        self.__initial_coefient=10
        
    def sigmoid(self, X, theta, intercept = 0 ):
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
    
    def lost(self,arg, y_target, x_i=1):
        """
        takes arg ,that is the result of sigmoid it has to be array
        y_target label variable which is 1 or 0
        x_i lement is every i element from X vectors in one array [x[i][j]] j is constant refer to column related to j_estimator
        """
        y = np.array(y_target)
        x_i = np.array(x_i)
        return (arg-y)*x_i
    
    def __cost(self,X, estimators, Y_label, intecept, x_i=1):
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
            raise ValueError('x_i and Y_label must have same shape')
            
        sigmoid_result = self.sigmoid(X, estimators, intecept)    
        result = self.lost(sigmoid_result, Y_label, x_i)
        return result.sum()   
    
    def gradient_descent(self):
        #x = np.array(X_data)
        #y_l = np.array(Y_label)
        s =len(self.X.shape)
        m = self.X.shape[0]
        
        if s>1:
            n = self.X.shape[1]
        else :
            n=1
        estimators = np.full(n, self.__initial_coefient)
        for i in range(self.__max_iter):
            for j in range(len(estimators)):
                x_column = self.X[:,j] if s>1 else self.X
                estimators[j]-=self.__cost(self.X, estimators, self.Y_label, self.intercept, x_column )*self.learning_rate

            self.intercept -=self.__cost(self.X, estimators, self.Y_label,  self.intercept, np.full(m, 1) )*self.learning_rate     
            self.coef=estimators
        
    
    def fit(self, X, y):
        
        """ 
        fit the model according to given data dataset 
        """
        self.Y_label=np.array(y)
        self.X=np.array(X)
        self.gradient_descent()
    
    def predict(self,X):
        """
        Predict class labels for samples in X.
        """
        
        sigmoid_estimator = self.sigmoid(X,self.coef)
        result = []
        for i in  sigmoid_estimator:
            if i < 0.5 :
                result.append(0)
            else :
                result.append(1)
        return np.array(result)     
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        """
        
        score = []
        values = self.predict(X)
        #if X.shape != y.shape:
          #  raise ValueError('X and y must be from same shape x={} y = {}'.format(X,y))
            
        for i in range(len(y)):
            if values[i]==y[i]:
                score.append(1)
            else:
                score.append(0)
        return(np.array(score).sum()/len(y))        
                
        
    
          
            
            
    def get_coef(self):
        return self.coef
    
    def get_intercept(self):
        return self.intercept
    
    def set_params(self,args):
        pass
 
 
    
    
    
    
    
#   TEST
        
    
z_f = lambda x : 5*x-4
x = np.linspace(-10,10,500)

z_args = np.array([z_f(i) for i in x])


y_prime = np.array( 1/(1+np.exp(-z_args)))


## generate labeled  from already define y_prime data given the sigmoid with args z = 2*x+4
def generated_label(i):
        if i < 0.3 :
            return 0
        elif i > 0.7 :
            return 1
        else :
            return np.random.randint(0,2)
        
 ####       

#####

#label = data[:,1] #.reshape(-1, 1)
#trained_data = data[:,0]
y_label = np.array([generated_label(i) for i in y_prime])



X_train, X_test, y_train, y_test = train_test_split(x, y_label, test_size=0.33, random_state=42)

br = BinaryLogisticRegression(learning_rate=0.01,max_iter=1000)   
rg = LogisticRegression(C=30)

br.fit(X_train,y_train)
rg.fit(X_train.reshape(-1, 1),y_train)

print(rg.coef_)
print(rg.intercept_)    

print(br.coef)
print(br.intercept) 
a = br.predict(X_test)
print(a)
print(br.score(X_test,y_test))
print(rg.score(X_test.reshape(-1, 1),y_test))
#print("Y______original",y_label)