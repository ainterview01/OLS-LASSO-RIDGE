#!/usr/bin/env python
# coding: utf-8

# In[93]:


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


# In[94]:


X, y = make_regression(n_samples=10001, n_features=5, noise=1, random_state=42)


# In[95]:


X=pd.DataFrame(X)
y=pd.DataFrame(y)


# In[97]:


X_fin=X.iloc[-1:]
y_fin=y.iloc[-1:]
X=X.iloc[:-1]
y=y.iloc[:-1]


# In[98]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[99]:


scaler = StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


# In[100]:


X_test

Let's test the OLS model first
# In[101]:


OLS= LinearRegression().fit(X_train, y_train)
y_pred_OLS=OLS.predict(X_test)


# In[102]:


mean_squared_error(y_pred_OLS, y_test)

Let's try the Ridge model now by using cross validation to find the hyperparameter
# In[105]:


Ridge = RidgeCV(alphas=[1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1,10,20,30,40,50,100]).fit(X_train, y_train)
y_pred_Ridge=Ridge.predict(X_test)


# In[107]:


mean_squared_error(y_pred_Ridge, y_test)


# Let's try the Lasso model now by using cross validation to find the hyperparameter

# In[108]:


Lasso = LassoCV(alphas=[1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1,10,20,30,40,50,100]).fit(X_train, y_train)
y_pred_Lasso=Lasso.predict(X_test)


# In[109]:


mean_squared_error(y_pred_Lasso, y_test)


# In[110]:


y_final_pred=Lasso.predict(X_fin)


# In[111]:


y_final_pred


# In[ ]:




