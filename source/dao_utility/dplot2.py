#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:36:04 2019

@author: daodeiv
"""
import matplotlib.pyplot as plt
plt.style.use('classic')
import seaborn as sns
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import math
from mpl_toolkits import mplot3d



class Dplot2D:
    def __init__(self, _range =10, fig_size=(10,10),elev=10,azim=10,mesh_const=0.1,**kwargs):
        self.fig_size = fig_size
        self.fig = plt.figure()
        self.fig = plt.figure(figsize=fig_size)
        self.ax = self.fig.gca()
       
        #self.ax = fig.add_subplot(111, projection='3d')
        #self.ax.set_aspect("equal")
        self.ax.set_xlabel('X',fontsize=30)
        #self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_ylabel('Y',fontsize=30)
        self.range=_range
        self.mesh = mesh_const
        #plt.xlabel('$\textbf{time} (s)$')
        
    def fucj(self):
        pass
    
    def append_funct(self,function,color='b',label=''):
        x,y,z = self._calculate_funct(function)
        self.ax.plot_wireframe(x, y,color=color,label=label,rstride=15, cstride=5)    
        
    def init_cordinates(self,xlimt=10,ylimt=10):
        
            self.ax.set_xlim([-xlimt, xlimt])
            self.ax.set_ylim([-ylimt, ylimt])
     
    
    def append_curve(self, y_funct,_range=5,color='y',label='',linewidth=1.0):
            """
            x= 2+k*2
            y = 3+k*4
            z = 5 + k*4
            """

            x = np.linspace(-_range,_range,100)
            y = y_funct(x)
            self.ax.plot(x, y,color=color,label=label,linewidth=linewidth)  
            
    def append_point(self,x,y,text=''):    
        self.ax.scatter(x,y)
        self.ax.text(x,y,text)
        
    def show(self):
        plt.legend()
        plt.show()   
        

class VectorGround:
    def __init__(self, range=[-3,10], fig_size=(20,20),**kwargs):
        self.plt = plt
        self.range=range
        self.ax = self.plt.gca()
        self.ax.set_aspect('equal')
        plt.grid()
        self.ax.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.1, fc='blue', ec='black')
        self.ax.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.1, fc='blue', ec='black')

        self.ax.text(1, -0.3, r'$\vec{e}_1$',fontsize=16,color='red')
        self.ax.text(-0.4, 1, r'$\vec{e}_2$',fontsize=16,color='red')
        self.plt.xlim(range[0],range[1])
        self.plt.ylim(range[0],range[1])
        self.ax.set_xlabel('X',fontsize=30)
        
        self.ax.set_ylabel('Y',fontsize=30)
        self.plt.title('',fontsize=10)

        self.plt.savefig('fig1.png', bbox_inches='tight')
    
    def add_v(self,x_0,y_0,x,y,index='1',show_cord=True,font_size=15):
        self.ax.arrow(x_0, y_0, x, y, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
        if show_cord:
            #self.ax.text(x, y-0.2, r'$\vec{r}_{%s}(%1.1f:%1.1f)$' % (index,x, y),fontsize=font_size,color='blue')
            self.ax.text(x, y-0.2, r'$\vec{r}_{%s}(%1.f:%1.f)$' % (index,x, y),fontsize=font_size,color='blue')
    


