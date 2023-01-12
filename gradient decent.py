#!/usr/bin/env python
# coding: utf-8

# In[6]:


import random
import os
import string

#to generate x data points

import numpy as np
l1=[]
n=1000
for i in range(n):
    l1.append(random.randint(1, 100))
   

#to generate y data points

import numpy as np
l2=[]
n=1000
for i in range(n):
    l2.append(random.randint(1, 100))


x=np.array(l1)
y=np.array(l2)
print(x)
print(y)






# In[7]:


x_mean=np.mean(x)
y_mean=np.mean(y)
x_mean,y_mean


# In[14]:


#to find slope and intercept

Sxy= np.sum(x*y)- n*x_mean*y_mean
Sxx= np.sum(x*x)-n*x_mean*x_mean
  
m = Sxy/Sxx
c = y_mean-b1*x_mean
print('slope m is', m)
print('intercept c is', c)
  


# In[10]:


#plotting the points

import matplotlib.pyplot as plt
plt.scatter(x,y)


# In[15]:


#linear regression model

plt.xlabel('Independent variable X')
plt.ylabel('Dependent variable y')
y_pred = m * x + c
  
plt.scatter(x, y, color = 'red')
plt.plot(x, y_pred, color = 'green')
plt.xlabel('X')
plt.ylabel('y')


# In[39]:


#sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X=np.array(x).reshape(-1,1)
Y=np.array(y).reshape(-1,1)
X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.2)
regressor=LinearRegression()
regressor.fit(X_train, Y_train)


print("slope,m is", regressor.coef_)
print("intercept is", regressor.intercept_)
    


# In[37]:


#gradient decent

m=0
c=0
lr=0.0001
iterations=10000
n=float(len(x))
for i in range(iterations):
    y_pred = m * x + c
    d_m=(-2/n)*sum(x*(y-y_pred))
    d_c=(-2/n)*sum(y-y_pred)
    m=m-lr*d_m
    c=c-lr*d_c
y_pred=m*x+c
print("slope m is",m)
print("intercept c is",c)
    


# In[ ]:




