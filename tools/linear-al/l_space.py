import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')



def cartesian_product(x,y):
    """
    retrun catesian product
    """
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])



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
    def plot_points(self,points,color='b'):
        for p in points:
            self.plt.scatter(p[0],p[1],color=color)
    def show(self):
        self.plt.show()

def linear_tranformation(points,matrix):
    sh = points.shape
    print(sh)
    A = np.array([0,0])
    for i in points[:][:]:
        row = matrix.dot(i)
        A=np.vstack([A, row])
    return A


x_args=y_args = np.linspace(0,10,9)
points = np.meshgrid(x_args,y_args)
result = cartesian_product(x_args,y_args)
print(type(result))

g = VectorGround(range=[-10,30])
#g.plot_points(result)
#new_points = linear_tranformation(result,np.array([[2,0],[0,1]]))  
#g.plot_points(new_points,color='r')
g.show()
#plt.show()
#new_points            


class VectorSpace:

    def __init__(self):
        pass

    def add_p(self):
        pass

    def add_vector(self):
        pass

class VectorSpace2D(VectorSpace):
    
    def __init__(self,range=[-3,10], fig_size=(20,20),title='a', **kwargs):
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



v = VectorSpace2D()
plt.show()