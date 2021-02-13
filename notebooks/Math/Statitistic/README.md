```python
def cartesian_product(x,y):
    """
    retrun catesian product
    """
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])



class VectorGround:
    def __init__(self, range=[-3,10], fig_size=(20,20),**kwargs):
        self.plt = plt
        self.plt.figure(figsize=fig_size)
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
        self.ax.set_xlabel('X',fontsize=20)
        
        self.ax.set_ylabel('Y',fontsize=30)
        self.plt.title('',fontsize=10)

        self.plt.savefig('fig1.png', bbox_inches='tight')
    
    def add_v(self,x_0,y_0,x,y,index='1',show_cord=True,font_size=15):
        self.ax.arrow(x_0, y_0, x, y, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
        if show_cord:
            #self.ax.text(x, y-0.2, r'$\vec{r}_{%s}(%1.1f:%1.1f)$' % (index,x, y),fontsize=font_size,color='blue')
            self.ax.text(x, y-0.2, r'$\vec{r}_{%s}(%1.0f:%1.0f)$' % (index,x, y),fontsize=font_size,color='blue')
    def plot_points(self,points,color='b'):
        for p in points:
            self.plt.scatter(p[0],p[1],color=color)
```


```python
import sys
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir =parentdir+'/'+'daoutil'
print( parentdir)
sys.path.insert(0,parentdir) 

import matplotlib.pyplot as plt
import numpy as np
import math

#my libs
from dplot import Dplot2D
```

    /home/daodeiv/GIT_PROJECTS/Machine-Learning/daoutil
    

### Eigenvalues and Eigenvectors

### deff:
### $$ 1)  \;\; Ax=\gamma x$$
### Where $A \in R^{n*n}$ matrix $ x \in R^n$ vector $\gamma \in R$ .the vector $x$ is einvector  $\gamma$  is eingvalue if $x$ and $\gamma$ is a solution of equation <br>
example : 
  $$2)  \;\; \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix} * \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}=  \begin{bmatrix} \gamma & 0\\ 0 & \gamma \end{bmatrix}* \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}= $$ <br>
  
   $$  \;\; \Big{(} \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}-\begin{bmatrix} \gamma & 0\\ 0 & \gamma \end{bmatrix} \Big{)} * \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}= \begin{bmatrix} 4-\gamma  & 2 \\ 1 & 3-\gamma  \end{bmatrix} * \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}=\begin{bmatrix} 0 \\ 0\end{bmatrix}$$ <br> 
   The homogens system has a solution only when <br>
$$ det \left(
\begin{array}{ccc}
   4-\gamma  & 2 \\ 1 & 3-\gamma 
\end{array} \right)=0$$ <br>

We factorize the characteristic polynomial and obtain  $$ p(λ) = (4 − λ)(3 − λ) − 2 · 1 = 10 − 7λ + λ
2 = (2 − λ)(5 − λ) $$ <br>

giving the roots $λ_1 = 2$ and $λ_2 = 5$ which are the *einvalues of matrix 1) <br> ,<br>
$$\begin{bmatrix} 4-\gamma  & 2 \\ 1 & 3-\gamma  \end{bmatrix} * \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}=\begin{bmatrix} 0 \\ 0\end{bmatrix}$$
For  $λ_2 = 2$  we obtain : <br> 
$$\begin{bmatrix} 4-2  & 2 \\ 1 & 3-2 \end{bmatrix} * \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}=\begin{bmatrix} 2  & 2 \\ 1 & 1  \end{bmatrix} * \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}=\begin{bmatrix} 0 \\ 0\end{bmatrix}$$ <br> <br> 
This means any vector  $x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} $, where x2 = −x1, such as $x=1$ and $x_2=-1$ 
, is an eigenvector with eigenvalue 2. The corresponding eigenspace is given as : <br>
$$ E_1 = span[\begin{bmatrix} 10 \\ -10 \end{bmatrix}]$$ <br> <br>

For  $λ_2 = 5$  we obtain : <br> 
$$\begin{bmatrix} 4-5  & 2 \\ 1 & 3-5 \end{bmatrix} * \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}=\begin{bmatrix} -1  & 2 \\ 1 & -2  \end{bmatrix} * \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}=\begin{bmatrix} 0 \\ 0\end{bmatrix}$$ <br> <br> 
We solve this homogeneous system and obtain a solution space :
$$E_2 = span[\begin{bmatrix} 20 \\ 10 \end{bmatrix}]$$ <br>

### let's to make the linear operator with einvalues E_1 and E_2 of matrix $ A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}$ <br> <br>
$$ T= \begin{bmatrix} 10 & 20 \\ -10 & 10 \end{bmatrix}$$

and to see the linear tranformation of point $x \in R^2$
## Graphical Intuition in Two Dimensions


```python
x_args=y_args = np.linspace(-5,5,9)
points = np.meshgrid(x_args,y_args)
result = cartesian_product(x_args,y_args)
print(type(result))

g = VectorGround(range=[-50,50])



def linear_tranformation(points,matrix):
    sh = points.shape
    print(sh)
    A = np.array([0,0])
    for i in points[:][:]:
        row = matrix.dot(i)
        A=np.vstack([A, row])
    return A

g.plot_points(result)
new_points = linear_tranformation(result,np.array([[4,2],[1,3]]))  
g.plot_points(new_points,color='r')
g.add_v(0,0,-1*10,1*10)
g.add_v(0,0,2*10,1*10)
```

    <class 'numpy.ndarray'>
    (81, 2)
    


![png](output_5_1.png)


## Theorems :

### Theorem 1. The eigenvectors $x_1, . . . , x_n$ of a matrix A ∈ Rn×n with n distinct eigenvalues λ_1, . . . , λ_n are linearly independent.
The proof comes from that the matrix ...

### Theorem 2. Given a matrix $ A ∈ R^{m×n} $ , we can always obtain a symmetric, positive semidefinite matrix $S ∈ R*{n×n} $ by defining
$$S := A^TA \;\; \;\; 1)$$
Understanding why Theorem 2 holds is insightful for how we can
    use symmetrized matrices: Symmetry requires $S = S^T$ , and by inserting (1) we obtain $S = A^TA=A^T(A^T)^T=(A^TA)=S^T$


### Theorem 3 (Spectral Theorem). If $A ∈ R^{n×n}$ is symmetric, there exists an orthonormal basis of the corresponding vector space $V$ consisting of eigenvectors of $A$, and each eigenvalue is real.


Example : <br>
Consider the matrix <br>
    $$ A= \begin{bmatrix} 3 & 2 & 2 \\ 2 & 3 & 2 \\ 2 & 2 & 3  \end{bmatrix}$$
    
The characteristic polynomial of $A$ is $$p_A(λ) = −(λ − 1)^2(λ − 7)$$
so that we obtain the eigenvalues $λ_1 = 1$ and $λ_2 = 7$, where $λ_1$ is a
repeated eigenvalue. Following our standard procedure for computing
eigenvectors, we obtain the eigenspaces : <br>
$$E_1 = span[\begin{bmatrix} -1 \\ 1 \\ 0 \end{bmatrix},\begin{bmatrix} -1 \\ 0\\ 1 \end{bmatrix}],\;\; E_7=span[\begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}]$$ <br> <br>
We see that $x_3$ is orthogonal to both $x_1$ and $x_2$. However, since $x_1^T x_2=1\neq 0$ they are not orthogonal. The spectral theorem (3)
states that there exists an orthogonal basis, but the one we have is not
orthogonal. However, we can construct one.
To construct such a basis, we exploit the fact that x1, x2 are eigenvectors associated with the same eigenvalue λ. Therefore, for any α, β ∈ R it
holds that <br> 
$$A(αx_1 + βx_2) = Ax_1α + Ax_2β = λ(αx_1 + βx_2)$$,<br> <br>

i.e., any linear combination of $x1$ and $x2$ is also an eigenvector of A associated with λ. The Gram-Schmidt algorithm is a method
for iteratively constructing an orthogonal/orthonormal basis from a set of
basis vectors using such linear combinations. Therefore, even if $x_1$ and $x_2$
are not orthogonal, we can apply the Gram-Schmidt algorithm and find
eigenvectors associated with $λ1 = 1$ that are orthogonal to each other (and to x_3). In our example, we will obtain
  
  
$$ x_1^{'}= \begin{bmatrix} -1 \\ 1 \\ 0 \end{bmatrix},\;\;   x_2^{'} = \frac{1}{2} \begin{bmatrix} -1 \\ -1\\ 2 \end{bmatrix}$$ <br> <br>  
which are orthogonal to each other, orthogonal to x_3, and eigenvectors of $A$ associated with λ_1 = 1.

### Decomposition
#### Cholesky Factorization


```python

```
