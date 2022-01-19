import pandas as pd
import collections
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../../../resources/data/500_Person_Gender_Height_Weight_Index.csv")
print(df.head())

df['obese'] = (df.Index > 4).astype('int')
df.drop('Index', axis=1, inplace=True)


def entropy(X, y):
    """

    """
    counter = collections.Counter(y)
    p = np.array(list(counter.values())) / len(y)
    return -np.sum(p * np.log2(p))



weights = df['Weight'].unique()
for i in range(1,len(weights)):
    weight = weights[i]
    a = df[df['Weight']<weight]
    e = entropy(a.drop('obese',axis=1),a['obese'])
    print(e)