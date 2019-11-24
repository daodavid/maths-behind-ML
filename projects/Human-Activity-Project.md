

```python
%matplotlib inline
```


```python
#import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib import rcParams, cycler

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
#import scikitplot as skplt
import datetime
```

# Human Activity Recognition with Smartphones
## Can smarthone predict our action ? <br>
### author David Stankov -  daodeiv

### Abstract

####  In the notebook we will try to show that the smartphones can predict our action by Support Vector Machine with accuracy approximately equal to 90%.Also, we will show the power of Principle Component Analysis as decrease dimension of our dataset from 563 columns to 25

### Introduction
#### The Human Activity Recognition database was built from the recordings of 30 study participants performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors. The objective is to classify activities into one of the six activities performed.Each person performed six activities wearing a smartphone (Samsung Galaxy S II) on the waist Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz.The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity.

### Data Load and  Exploration


```python
train = pd.read_csv('https://disk.bg/s/UFQtiaqlTDj2SZX/download')  # train data 
test = pd.read_csv('https://disk.bg/s/VYmtGL4lzcHdcCl/download')   # test data
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tBodyAcc-mean()-X</th>
      <th>tBodyAcc-mean()-Y</th>
      <th>tBodyAcc-mean()-Z</th>
      <th>tBodyAcc-std()-X</th>
      <th>tBodyAcc-std()-Y</th>
      <th>tBodyAcc-std()-Z</th>
      <th>tBodyAcc-mad()-X</th>
      <th>tBodyAcc-mad()-Y</th>
      <th>tBodyAcc-mad()-Z</th>
      <th>tBodyAcc-max()-X</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag-kurtosis()</th>
      <th>angle(tBodyAccMean,gravity)</th>
      <th>angle(tBodyAccJerkMean),gravityMean)</th>
      <th>angle(tBodyGyroMean,gravityMean)</th>
      <th>angle(tBodyGyroJerkMean,gravityMean)</th>
      <th>angle(X,gravityMean)</th>
      <th>angle(Y,gravityMean)</th>
      <th>angle(Z,gravityMean)</th>
      <th>subject</th>
      <th>Activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.288585</td>
      <td>-0.020294</td>
      <td>-0.132905</td>
      <td>-0.995279</td>
      <td>-0.983111</td>
      <td>-0.913526</td>
      <td>-0.995112</td>
      <td>-0.983185</td>
      <td>-0.923527</td>
      <td>-0.934724</td>
      <td>...</td>
      <td>-0.710304</td>
      <td>-0.112754</td>
      <td>0.030400</td>
      <td>-0.464761</td>
      <td>-0.018446</td>
      <td>-0.841247</td>
      <td>0.179941</td>
      <td>-0.058627</td>
      <td>1</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.278419</td>
      <td>-0.016411</td>
      <td>-0.123520</td>
      <td>-0.998245</td>
      <td>-0.975300</td>
      <td>-0.960322</td>
      <td>-0.998807</td>
      <td>-0.974914</td>
      <td>-0.957686</td>
      <td>-0.943068</td>
      <td>...</td>
      <td>-0.861499</td>
      <td>0.053477</td>
      <td>-0.007435</td>
      <td>-0.732626</td>
      <td>0.703511</td>
      <td>-0.844788</td>
      <td>0.180289</td>
      <td>-0.054317</td>
      <td>1</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.279653</td>
      <td>-0.019467</td>
      <td>-0.113462</td>
      <td>-0.995380</td>
      <td>-0.967187</td>
      <td>-0.978944</td>
      <td>-0.996520</td>
      <td>-0.963668</td>
      <td>-0.977469</td>
      <td>-0.938692</td>
      <td>...</td>
      <td>-0.760104</td>
      <td>-0.118559</td>
      <td>0.177899</td>
      <td>0.100699</td>
      <td>0.808529</td>
      <td>-0.848933</td>
      <td>0.180637</td>
      <td>-0.049118</td>
      <td>1</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.279174</td>
      <td>-0.026201</td>
      <td>-0.123283</td>
      <td>-0.996091</td>
      <td>-0.983403</td>
      <td>-0.990675</td>
      <td>-0.997099</td>
      <td>-0.982750</td>
      <td>-0.989302</td>
      <td>-0.938692</td>
      <td>...</td>
      <td>-0.482845</td>
      <td>-0.036788</td>
      <td>-0.012892</td>
      <td>0.640011</td>
      <td>-0.485366</td>
      <td>-0.848649</td>
      <td>0.181935</td>
      <td>-0.047663</td>
      <td>1</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.276629</td>
      <td>-0.016570</td>
      <td>-0.115362</td>
      <td>-0.998139</td>
      <td>-0.980817</td>
      <td>-0.990482</td>
      <td>-0.998321</td>
      <td>-0.979672</td>
      <td>-0.990441</td>
      <td>-0.942469</td>
      <td>...</td>
      <td>-0.699205</td>
      <td>0.123320</td>
      <td>0.122542</td>
      <td>0.693578</td>
      <td>-0.615971</td>
      <td>-0.847865</td>
      <td>0.185151</td>
      <td>-0.043892</td>
      <td>1</td>
      <td>STANDING</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 563 columns</p>
</div>




```python
train.shape
```




    (7352, 563)




```python
# separating data inputs and output lables 
train_data  = train.drop('Activity' , axis=1).values
label_data = train['Activity'].values

test_data = test.drop('Activity' , axis=1).values
test_label = test['Activity'].values


print('train data shape ' ,train_data.shape)
print('label data shape' ,label_data.shape)

print('train data test shape' ,test_data_.shape)
print('label_data_test shape' ,label_test_train.shape)
```

    train data shape  (7352, 562)
    label data shape (7352,)
    train data test shape (2947, 562)
    label_data_test shape (2947,)



```python
activity = np.unique(label_data)
activity
```




    array(['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS',
           'WALKING_UPSTAIRS'], dtype=object)



### Feature Selection

#### Since some of feature maybe depends each other  are unnecessary.Others have a small variance and they unnecessary as  well.Before  we apply dimensionality reduction over data sets,will be bettet to scalling data, The origin data  maybe is scalled enough nevertheless let's to apply standart scaller

#### The statistical Description before Scalling


```python
pd.DataFrame(train_data).describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>552</th>
      <th>553</th>
      <th>554</th>
      <th>555</th>
      <th>556</th>
      <th>557</th>
      <th>558</th>
      <th>559</th>
      <th>560</th>
      <th>561</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>...</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
      <td>7352.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.274488</td>
      <td>-0.017695</td>
      <td>-0.109141</td>
      <td>-0.605438</td>
      <td>-0.510938</td>
      <td>-0.604754</td>
      <td>-0.630512</td>
      <td>-0.526907</td>
      <td>-0.606150</td>
      <td>-0.468604</td>
      <td>...</td>
      <td>-0.307009</td>
      <td>-0.625294</td>
      <td>0.008684</td>
      <td>0.002186</td>
      <td>0.008726</td>
      <td>-0.005981</td>
      <td>-0.489547</td>
      <td>0.058593</td>
      <td>-0.056515</td>
      <td>17.413085</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.070261</td>
      <td>0.040811</td>
      <td>0.056635</td>
      <td>0.448734</td>
      <td>0.502645</td>
      <td>0.418687</td>
      <td>0.424073</td>
      <td>0.485942</td>
      <td>0.414122</td>
      <td>0.544547</td>
      <td>...</td>
      <td>0.321011</td>
      <td>0.307584</td>
      <td>0.336787</td>
      <td>0.448306</td>
      <td>0.608303</td>
      <td>0.477975</td>
      <td>0.511807</td>
      <td>0.297480</td>
      <td>0.279122</td>
      <td>8.975143</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-0.999873</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>...</td>
      <td>-0.995357</td>
      <td>-0.999765</td>
      <td>-0.976580</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.262975</td>
      <td>-0.024863</td>
      <td>-0.120993</td>
      <td>-0.992754</td>
      <td>-0.978129</td>
      <td>-0.980233</td>
      <td>-0.993591</td>
      <td>-0.978162</td>
      <td>-0.980251</td>
      <td>-0.936219</td>
      <td>...</td>
      <td>-0.542602</td>
      <td>-0.845573</td>
      <td>-0.121527</td>
      <td>-0.289549</td>
      <td>-0.482273</td>
      <td>-0.376341</td>
      <td>-0.812065</td>
      <td>-0.017885</td>
      <td>-0.143414</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.277193</td>
      <td>-0.017219</td>
      <td>-0.108676</td>
      <td>-0.946196</td>
      <td>-0.851897</td>
      <td>-0.859365</td>
      <td>-0.950709</td>
      <td>-0.857328</td>
      <td>-0.857143</td>
      <td>-0.881637</td>
      <td>...</td>
      <td>-0.343685</td>
      <td>-0.711692</td>
      <td>0.009509</td>
      <td>0.008943</td>
      <td>0.008735</td>
      <td>-0.000368</td>
      <td>-0.709417</td>
      <td>0.182071</td>
      <td>0.003181</td>
      <td>19.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.288461</td>
      <td>-0.010783</td>
      <td>-0.097794</td>
      <td>-0.242813</td>
      <td>-0.034231</td>
      <td>-0.262415</td>
      <td>-0.292680</td>
      <td>-0.066701</td>
      <td>-0.265671</td>
      <td>-0.017129</td>
      <td>...</td>
      <td>-0.126979</td>
      <td>-0.503878</td>
      <td>0.150865</td>
      <td>0.292861</td>
      <td>0.506187</td>
      <td>0.359368</td>
      <td>-0.509079</td>
      <td>0.248353</td>
      <td>0.107659</td>
      <td>26.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.916238</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.967664</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.989538</td>
      <td>0.956845</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.998702</td>
      <td>0.996078</td>
      <td>1.000000</td>
      <td>0.478157</td>
      <td>1.000000</td>
      <td>30.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 562 columns</p>
</div>




```python
###applying standart scaller over test and train data
scaler = StandardScaler()
scaler.fit(train_data)
train_data = a.transform(train_data)
scaler.fit(test_data)
test_data = a.transform(test_data)
```

### The statistical Description after Scalling



```python
pd.DataFrame(train_data).describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>552</th>
      <th>553</th>
      <th>554</th>
      <th>555</th>
      <th>556</th>
      <th>557</th>
      <th>558</th>
      <th>559</th>
      <th>560</th>
      <th>561</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>...</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
      <td>7.352000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-3.166975e-16</td>
      <td>3.384504e-17</td>
      <td>9.365619e-17</td>
      <td>-1.226198e-16</td>
      <td>-1.977472e-16</td>
      <td>1.596474e-16</td>
      <td>7.520281e-18</td>
      <td>1.987287e-16</td>
      <td>1.078209e-16</td>
      <td>1.359087e-17</td>
      <td>...</td>
      <td>-7.505558e-17</td>
      <td>3.100077e-16</td>
      <td>-1.525198e-18</td>
      <td>5.899287e-18</td>
      <td>-1.418547e-17</td>
      <td>7.106515e-17</td>
      <td>1.277089e-16</td>
      <td>-1.075189e-17</td>
      <td>-2.398788e-17</td>
      <td>-4.450073e-15</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>...</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
      <td>1.000068e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.814049e+01</td>
      <td>-2.407152e+01</td>
      <td>-1.573085e+01</td>
      <td>-8.793362e-01</td>
      <td>-9.727918e-01</td>
      <td>-9.440787e-01</td>
      <td>-8.713436e-01</td>
      <td>-9.736247e-01</td>
      <td>-9.511122e-01</td>
      <td>-9.759168e-01</td>
      <td>...</td>
      <td>-2.144460e+00</td>
      <td>-1.217541e+00</td>
      <td>-2.925683e+00</td>
      <td>-2.235646e+00</td>
      <td>-1.658375e+00</td>
      <td>-2.079788e+00</td>
      <td>-9.974223e-01</td>
      <td>-3.558777e+00</td>
      <td>-3.380417e+00</td>
      <td>-1.828851e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.638693e-01</td>
      <td>-1.756427e-01</td>
      <td>-2.092798e-01</td>
      <td>-8.631868e-01</td>
      <td>-9.295295e-01</td>
      <td>-8.968638e-01</td>
      <td>-8.562288e-01</td>
      <td>-9.286823e-01</td>
      <td>-9.034203e-01</td>
      <td>-8.587814e-01</td>
      <td>...</td>
      <td>-7.339585e-01</td>
      <td>-7.162086e-01</td>
      <td>-3.866541e-01</td>
      <td>-6.507937e-01</td>
      <td>-8.072166e-01</td>
      <td>-7.749059e-01</td>
      <td>-6.301983e-01</td>
      <td>-2.571022e-01</td>
      <td>-3.113525e-01</td>
      <td>-1.048866e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.850502e-02</td>
      <td>1.167141e-02</td>
      <td>8.206943e-03</td>
      <td>-7.594273e-01</td>
      <td>-6.783764e-01</td>
      <td>-6.081593e-01</td>
      <td>-7.551036e-01</td>
      <td>-6.800064e-01</td>
      <td>-6.061248e-01</td>
      <td>-7.585419e-01</td>
      <td>...</td>
      <td>-1.142597e-01</td>
      <td>-2.809130e-01</td>
      <td>2.451015e-03</td>
      <td>1.507318e-02</td>
      <td>1.477537e-05</td>
      <td>1.174412e-02</td>
      <td>-4.296252e-01</td>
      <td>4.151086e-01</td>
      <td>2.138846e-01</td>
      <td>1.768243e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.988854e-01</td>
      <td>1.693906e-01</td>
      <td>2.003738e-01</td>
      <td>8.081625e-01</td>
      <td>9.484617e-01</td>
      <td>8.177048e-01</td>
      <td>7.966913e-01</td>
      <td>9.471052e-01</td>
      <td>8.222278e-01</td>
      <td>8.291412e-01</td>
      <td>...</td>
      <td>5.608611e-01</td>
      <td>3.947685e-01</td>
      <td>4.222002e-01</td>
      <td>6.484291e-01</td>
      <td>8.178415e-01</td>
      <td>7.644196e-01</td>
      <td>-3.816570e-02</td>
      <td>6.379332e-01</td>
      <td>5.882181e-01</td>
      <td>9.568092e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.032661e+01</td>
      <td>2.493878e+01</td>
      <td>1.958529e+01</td>
      <td>3.577947e+00</td>
      <td>2.839526e+00</td>
      <td>3.833088e+00</td>
      <td>3.845150e+00</td>
      <td>3.075829e+00</td>
      <td>3.878713e+00</td>
      <td>2.697113e+00</td>
      <td>...</td>
      <td>4.039226e+00</td>
      <td>5.144111e+00</td>
      <td>2.943656e+00</td>
      <td>2.225893e+00</td>
      <td>1.627550e+00</td>
      <td>2.096610e+00</td>
      <td>2.910568e+00</td>
      <td>1.410491e+00</td>
      <td>3.785390e+00</td>
      <td>1.402515e+00</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 562 columns</p>
</div>



#### Principle Component Analysis (PCA)


```python
pca = PCA()
pca.fit(train_data)
```




    PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)




```python
explained_variance = pca.explained_variance_
explained_variance[0:100]
```




    array([284.9303245 ,  36.92434577,  15.75073873,  14.04908594,
            10.59495967,   9.68544219,   7.69311009,   6.76026306,
             5.60228319,   5.41511781,   4.82719085,   4.49415468,
             4.31193538,   3.63933007,   3.55946718,   3.36519066,
             3.29654827,   3.22847767,   3.18970865,   2.9588797 ,
             2.80980896,   2.74079369,   2.71125901,   2.63313858,
             2.52956399,   2.37408006,   2.34830341,   2.28449435,
             2.19636391,   2.1723134 ,   2.05615542,   1.98962107,
             1.9664892 ,   1.89276521,   1.86209602,   1.84196575,
             1.80737671,   1.65891109,   1.61500408,   1.60016979,
             1.5294307 ,   1.49096019,   1.47791107,   1.45307239,
             1.40087449,   1.3856098 ,   1.35116808,   1.32972514,
             1.30907024,   1.27755458,   1.24354645,   1.20673863,
             1.16538397,   1.13925345,   1.12133709,   1.11596533,
             1.09051547,   1.07380237,   1.0626446 ,   1.0449445 ,
             1.02079759,   0.99623427,   0.98921253,   0.97250283,
             0.96181398,   0.95594758,   0.93496855,   0.91570091,
             0.89343576,   0.8842914 ,   0.86331031,   0.85734505,
             0.8317076 ,   0.82188287,   0.81931401,   0.81157797,
             0.7805549 ,   0.76690699,   0.75755101,   0.74695956,
             0.74039212,   0.73147692,   0.72227295,   0.70161351,
             0.69378412,   0.68259771,   0.67021332,   0.65945705,
             0.6556873 ,   0.63879018,   0.63484356,   0.61640137,
             0.61340321,   0.60347314,   0.5849916 ,   0.57497037,
             0.57009568,   0.56507711,   0.54847005,   0.54130722])




```python
print('min value of variance ' ,explained_variance.min())
print('max value of variance ' ,explained_variance.max())
print('std of variance array ' ,explained_variance.std())
print('count of variance array ' ,explained_variance.size)
```

    min value of variance  1.64066611022484e-32
    max value of variance  284.9303245033929
    std of variance array  12.160217531518953
    count of variance array  562


#### Let's  visualize the data reduced 3 dimensions by PCA 


```python
tranformed_train_data = pca.transform(train_data) # transform data by pca
tranformed_data_test= pca.transform(test_data)

tranformed_train_data = tranformed_train_data[:,:3]  ### get first 3 component analysis
tranformed_data_test = tranformed_data_test[:,:3]

activity = np.unique(label_data)
activity
```




    array(['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS',
           'WALKING_UPSTAIRS'], dtype=object)




```python
def activity_to_colors(activity):
    """
    mapping between activity and color
    """
    if activity=='LAYING':
        return 'yellow'
    elif activity=='SITTING':
        return 'blue' 
    elif activity=='STANDING':
        return 'red'  
    elif activity=='WALKING':
        return 'pink' 
    elif activity=='WALKING_DOWNSTAIRS':
        return 'purple' 
    elif activity=='WALKING_UPSTAIRS':
        return 'gray'
```


```python
def visialise_data(tranformed_train_data,colors):
    """
    3d scatter ploting with color associate label
    """
    fig = plt.figure(figsize=[20,20])
    ax = fig.add_subplot(111, projection='3d')
    lines = ax.scatter(tranformed_train_data[:,0], tranformed_train_data[:,1], tranformed_train_data[:,2],c = colors,\
           alpha=0.5,edgecolors='none')

    yellow_patch = mpatches.Patch(color='yellow', label='LAYING')
    blue_patch = mpatches.Patch(color='blue', label='SITTING')
    red_patch = mpatches.Patch(color='red', label='STANDING')
    green_pink = mpatches.Patch(color='pink', label='WALKING')
    purple_patch = mpatches.Patch(color='purple', label='WALKING_DOWNSTAIRS')
    gray_patch = mpatches.Patch(color='gray', label='WALKING_UPSTAIRS')

    plt.legend(handles=[yellow_patch, blue_patch,red_patch,green_pink,purple_patch,gray_patch])
```


```python
colors = np.array([activity_to_colors(i)  for i in label_data])
visialise_data(tranformed_train_data,colors)
```


![png](output_27_0.png)


####  We can see very cleared that, the data with only 3 component analysis has very good clustered representation. This brings the sense that one classification model can make a good job as SVM <br>

#### Let's  investigate components to find the smallest count of component which are the most important and can describe the data, without losing the necessary info


```python
plt.hist(explained_variance,bins=40)
plt.xlabel('variance of component')
plt.ylabel('distribution of a component of variance')
plt.title('The volume of a component of variance')
```




    Text(0.5, 1.0, 'The volume of a component of variance')




![png](output_30_1.png)



```python
def description_variance(limit):
    """
    limit -the limit of distribution
    
    """
    explained_variancel_less_then_10 = explained_variance[explained_variance<limit]
    plt.hist(explained_variancel_less_then_10,bins=40)
    plt.xlabel('The volume of a component of variance')
    plt.ylabel('distribution of a component of variance')

    plt.title('distribution of a component of variance with a volume less than '+str(limit))
    size = explained_variancel_less_then_10.size
    print('The volume of a component of variance less than '+str(limit) + ' are',size)  
    size = explained_variance[explained_variance>limit].size 
    print('The volume of a component of variance more than '+str(limit) + ' are',size)  
    plt.show()                    

        

    
   
```

#### Let's to see distributions of variance of component ,when variaces are less the 50,10,5,0.5,0.2


```python
for i in [50,10,5] :
    description_variance(i) 
```

    The volume of a component of variance less than 50 are 561
    The volume of a component of variance more than 50 are 1



![png](output_33_1.png)


    The volume of a component of variance less than 10 are 557
    The volume of a component of variance more than 10 are 5



![png](output_33_3.png)


    The volume of a component of variance less than 5 are 552
    The volume of a component of variance more than 5 are 10



![png](output_33_5.png)


#### from above graphics distribution ,maybe the significant component is the first components more then approsimatly 2.5


```python
description_variance(2.5) 
```

    The volume of a component of variance less than 2.5 are 537
    The volume of a component of variance more than 2.5 are 25



![png](output_35_1.png)


### The volume of a component of variance more than 2.5 are 25,we will train our model only over these components.because the component with  2.5 << from the first component with value 284.  For now, we will  reduce the number of components from 562 to 25 , which is a significant reduction <br>

#### let to get the tranform our data with new basic of covariance,a to get the first 25 columns as our new data <br>


```python
tranformed_train_data = pca.transform(train_data) # transform data by pca
tranformed_data_test= pca.transform(test_data)

tranformed_train_data = tranformed_train_data[:,:25]  ### get first 3 component analysis
tranformed_data_test = tranformed_data_test[:,:25]
print('shape tranformed train data ',tranformed_train_data.shape)
```

    shape tranformed train data  (7352, 25)


### we've reduced the count of columns from 563 to 25

### Predictive Modelling
#### To train we will use Suppor Vector Mashime


```python
svc = LinearSVC(C=1e9)
svc.fit(tranformed_train_data,label_data)
```

    /home/daodeiv/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)





    LinearSVC(C=1000000000.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
         verbose=0)




```python
svc.score(tranformed_data_test,test_label)
```




    0.8710553104852392




###  With linear SVM we achieve $0.85$ acuurancy.Let to try what will happen with  radiues base kernel !
<br> <br>


```python
kernel_svc = SVC(kernel="rbf")
kernel_svc.fit(tranformed_train_data,label_data)
```

    /home/daodeiv/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)





    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)




```python
kernel_svc.score(tranformed_data_test,test_label)
```




    0.7750254496097726



#### With radius base kernel SVM we achieve $0.77$ accuracy.Which is more bad then linear, but we didn't use the tuning of hyperparameters. In the next section we will try to improve our model by tuning hyperparameters <br>

### Tuning HyperParameter (SVM)


```python
param_grid = {'C':[100,1000],'kernel':['linear','rbf'],'gamma':[0.0001,0.00001]} ##  'kernel':['linear','rbf'] 'gamma':[1,0.1,0.001,0.0001]
grid = GridSearchCV(SVC(kernel="rbf"),param_grid)
grid.fit(tranformed_train_data,label_data)

```


```python
grid.best_params_
```




    {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}



#### Our greed search shows that the best parameters are C=100 and gamma = 0.0001, but our test is made with less variants , If we put more variants the grid search will find better tuning, unfortunately, the process is very slow


```python
kernel_svc = SVC(kernel="rbf",C=100,gamma=0.0001)
kernel_svc.fit(tranformed_train_data,label_data)
kernel_svc.score(tranformed_data_test,test_label)
```




    0.8883610451306413




```python
kernel_svc = SVC(kernel="rbf",C=1e9,gamma=0.0001)
kernel_svc.fit(tranformed_train_data,label_data)
kernel_svc.score(tranformed_data_test,test_label)
```


```python
kernel_svc = SVC(kernel="rbf",C=1e9)
kernel_svc.fit(tranformed_train_data,label_data)
kernel_svc.score(tranformed_data_test,test_label)
```

### Conclusion
#### We achieve 0.88 accuracies,and we can say by the result, The smartphone can preduct our action,also we have successfully to reduced by PCA our columns from 653 to 25  saving   the most importent feature     of dataset, which is good performance and featuring selection <br>
### External links
[1] https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones <br>
[2] https://softuni.bg/trainings/2317/machine-learning-september-2019 <br>
[3] https://datascienceplus.com/understanding-the-covariance-matrix/ <br>
[4] https://github.com/Daodavid93/Machine-Learning/blob/Principle_compenent_analysis/math/Eigendecomposition%20of%20a%20covariance%20matrix.ipynb <br>
[5] https://github.com/Daodavid93/Machine-Learning/tree/Principle_compenent_analysis
