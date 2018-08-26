
# coding: utf-8

# In[1]:


import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
from time import time
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import cross_validation, metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score , classification_report, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import *
from sklearn import metrics


# In[2]:


csv_filename="AirQualityUCI.csv"
df=pd.read_csv(csv_filename, sep=";" , parse_dates= ['Date','Time'])


# In[5]:


df.head()


# In[4]:


df.dropna(how="all",axis=1,inplace=True)


# In[6]:


df.shape


# In[7]:


df = df[:9357]


# In[8]:


df.tail()


# In[9]:


cols = list(df.columns[2:])


# In[10]:


for col in cols:
    if df[col].dtype != 'float64':
        str_x = pd.Series(df[col]).str.replace(',','.')
        float_X = []
        for value in str_x.values:
            fv = float(value)
            float_X.append(fv)

            df[col] = pd.DataFrame(float_X)

df.head()


# In[11]:


features=list(df.columns)


# In[12]:


features.remove('Date')
features.remove('Time')
features.remove('PT08.S4(NO2)')


# In[13]:


X = df[features]
y = df['C6H6(GT)']


# In[14]:


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.4, random_state=0)


# In[15]:


print(X_train.shape, y_train.shape)

