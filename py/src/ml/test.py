import pandas as pd
import collections
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../../../resources/data/500_Person_Gender_Height_Weight_Index.csv")
print(df.head())

df['obese'] = (df.Index > 4).astype('int')
df.drop('Index', axis=1, inplace=True)

a = df[df['Weight'] > 80]['obese'] == 1

print(len(a))
print(sum(a))
print(sum(a) / len(a))


def entropy(y):
    """

    """
    counter = collections.Counter(y)
    p = np.array(list(counter.values())) / len(y)
    return -np.sum(p * np.log2(p))

weights = df['Weight'].unique()
data = np.array([0,0])
for i in range(1,len(weights)):
    weight = weights[i]
    a = df[df['Weight']<weight]
    e = entropy(a['obese'])
    data = np.vstack([data,[weight,e]])

plt.scatter(data[:,0],data[:,1],s=4)
#plt.show()
def inf_gain(y,mask,func=entropy):
    y_left = y[mask]
    y_right = y[-mask]

    ig = func(y)-(len(y_left)/len(y)*func(y_left) + len(y_right)/len(y)*func(y_right))