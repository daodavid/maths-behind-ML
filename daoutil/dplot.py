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
    
    


