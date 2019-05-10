#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import required libraries
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn import svm ,datasets
from sklearn.model_selection import train_test_split


# In[3]:


#importing iris dataset
iris=datasets.load_iris()


# In[4]:


#viewing features and labels
iris.data
iris.target


# In[5]:


#converting np arrsy of iris dataset to pandas dataframe
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df["class"]=iris.target
df


# In[6]:


#dividing the data and labels
#where x is the feature variable and y is the label that needs to be predicted
x=df.drop(columns="class")
y=df["class"]


# In[7]:


#split data to train and test
x_train,x_test,y_train,y_test=train_test_split(x,y)


# In[50]:


#using svm model from sklearn and fitting the data
model_rbf=svm.SVC(C=1.0,gamma="auto", kernel="rbf")


# In[51]:


model_rbf.fit(x_train,y_train)


# In[52]:


model_rbf.score(x_test,y_test)


# In[ ]:




