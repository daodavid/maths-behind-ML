import matplotlib.pyplot as plt
import numpy as np

# 

def v_plot(X_1,X_2,**kwargs):
    '''
    X,Y : nd.array,int
    
    
    '''
    fig, ax = plt.subplots()
    plt.grid()
    ax.set_aspect('equal')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    if isinstance(X_1, (list,tuple, np.ndarray)):
        x_0=y_0 = np.zeros(len(X_1))
    else :
        x_0=y_0=0

      #print(len(x_1))
    q = ax.quiver(x_0, y_0, X_1, X_2,**kwargs)

v_plot(1,4,units='xy' ,scale=1,color='r',label='wa')   
plt.show()