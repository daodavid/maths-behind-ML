# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:42:18 2020

@author: David
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
#define z argument as lininear euation of z = 2*x + 4
z_f = lambda x : 2*x
x = np.linspace(-20,20,8400)

z_args = np.array([z_f(i) for i in x])

#print(z_args)


def sigmoid(z):
    result = [1/(1+np.exp(-i)) for i in z ]
    return result

y_prime = np.array( sigmoid(z_args))

print(y_prime.shape)
print(z_args.shape)
print(x.shape)


## generate labeled  from already define y_prime data given the sigmoid with args z = 2*x+4
def generated_label(i):
        if i < 0.5 :
            return 0
        elif i > 0.5 :
            return 1
        else :
            return np.random.randint(0,2)
        
        
        
y_label = np.array([generated_label(i) for i in y_prime])        


plt.plot(z_args,y_prime)
plt.scatter(z_args,y_label)
plt.scatter(z_args,y_prime,color='red')

data = np.array([z_args,y_label]).T




def sigmoid(a,b,x_data):
  
    return 1/(1.0+np.exp(-(a*x_data+b)))

def cost_a(a,b,x_data,y_data):
    g = sigmoid(a,b,x_data)
    p = g-y_data
    p =p*x_data
    p = p.sum()/len(x_data)
    return p
    
    #return (1/len(x_data))*((y_data-sigmoid(a,b,x_data))*x_data).sum()
     
    
def cost_b(a,b,x_data,y_data):
    g = sigmoid(a,b,x_data)
    p = g-y_data
    p = p.sum()/len(x_data)
    return p   
    #return (1/len(x_data))*((y_data-sigmoid(a,b,x_data))).sum()
    


def perform_gradient_descent(x_data,y_data,times_interaction=43999,learning_rate=0.001):
    a = -10
    b = 0
    a_args=[a]
    b_args=[b]
    for i in range(times_interaction):
        j_1 = cost_a(a,0,x_data,y_data)
        #j_2 = cost_b(a,0,x_data,y_data)
        a = a -j_1*learning_rate
        #b = b -j_2*learning_rate
        a_args=np.append(a_args, a)
        b_args=np.append(b_args, b)
    print(a,b)
    return a,b,a_args,b_args


label = data[:,1] #.reshape(-1, 1)
trained_data = data[:,0]
reg = LogisticRegression(C=1,fit_intercept=False)
reg.fit(trained_data.reshape(-1, 1),label)
print("a = ",reg.coef_)
print("b=",reg.intercept_)
perform_gradient_descent(trained_data,label)