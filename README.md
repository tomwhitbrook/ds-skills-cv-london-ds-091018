
# Cross Validation

Cross validation is very useful for determining optimal model parameters such as our regularization parameter alpha. It first divides the training set into subsets (by default the sklearn package uses 3) and then selects an optimal hyperparameter (in this case alpha, our regularization parameter) based on test performance. For example, if we have 3 splits: A, B and C, we can do 3 training and testing combinations and then average test performance as an overall estimate of model performance for those given parameters. (The three combinations are: Train on A+B test on c, train on A+C test on B, train on B+C test on A.) We can do this across various alpha values in order to determine an optimal regularization parameter. By default, sklearn will even estimate potential alpha for you, or you can explicit check the performance of specific alpha.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
```


```python
df = pd.read_csv('Housing_Prices/train.csv')
print(len(df))
df.head()
```

    1460





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
from sklearn.linear_model import LassoCV, RidgeCV
```


```python
#Define X and Y
feats = [col for col in df.columns if df[col].dtype in [np.int64, np.float64]]

X = df[feats].drop('SalePrice', axis=1)

#Impute null values
for col in X:
    avg = X[col].mean()
    X[col] = X[col].fillna(value=avg)

y = df.SalePrice

print('Number of X features: {}'.format(len(X.columns)))

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y)
L1 = LassoCV()
print('Model Details:\n', L1)

L1.fit(X_train, y_train)

print('Optimal alpha: {}'.format(L1.alpha_))
print('First 5 coefficients:\n', L1.coef_[:5])
count = 0
for num in L1.coef_:
    if num == 0:
        count += 1
print(count)
print('Number of coefficients set to zero: {}'.format(count))
```

    Number of X features: 37
    Model Details:
     LassoCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
        max_iter=1000, n_alphas=100, n_jobs=1, normalize=False, positive=False,
        precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
        verbose=False)
    Optimal alpha: 198489.80980228688
    First 5 coefficients:
     [-2.80735194 -0.         -0.          0.25507382  0.        ]
    25
    Number of coefficients set to zero: 25


# Notes on Coefficients and Using Lasso for Feature Selection
The Lasso technique also has a very important and profound effect: feature selection. That is, many of your feature coefficients will be optimized to zero, effectively removing their impact on the model. This can be a useful application in practice when trying to reduce the number of features in the model. Note that which variables are set to zero can change if multicollinearity is present in the data. That is, if two features within the X space are highly correlated, then which takes precendence in the model is somewhat arbitrary, and as such, coefficient weights between multiple runs of `.fit()` could lead to substantially different coefficient values.

# With Normalization


```python
#Define X and Y
feats = [col for col in df.columns if df[col].dtype in [np.int64, np.float64]]

X = df[feats].drop('SalePrice', axis=1)

#Impute null values
for col in X:
    avg = X[col].mean()
    X[col] = X[col].fillna(value=avg)

y = df.SalePrice

print('Number of X features: {}'.format(len(X.columns)))

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y)
L1 = LassoCV(normalize = True)
print('Model Details:\n', L1)
L1.fit(X_train, y_train)

print('Optimal alpha: {}'.format(L1.alpha_))
print('First 5 coefficients:\n', L1.coef_[:5])
count = 0
for num in L1.coef_:
    if num == 0:
        count += 1
print(count)
print('Number of coefficients set to zero: {}'.format(count))
```

    Number of X features: 37
    Model Details:
     LassoCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
        max_iter=1000, n_alphas=100, n_jobs=1, normalize=True, positive=False,
        precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
        verbose=False)
    Optimal alpha: 141.0984264427501
    First 5 coefficients:
     [ -0.00000000e+00  -5.95275404e+01   0.00000000e+00   1.60217484e-01
       2.00649624e+04]
    21
    Number of coefficients set to zero: 21


# Calculate the Mean Squarred Error 
Calculate the mean squarred error between both of the models above and the test set.


```python
# Your code here
```

# Repeat this Process for the Ridge Regression Object


```python
# Your code here
```

# Practice Preprocessing and Feature Engineering
Use some of our previous techniques including normalization, feature engineering, and dummy variables on the dataset. Then, repeat fitting and tuning a model, observing the performance impact compared to above.


```python
# Your code here
```
