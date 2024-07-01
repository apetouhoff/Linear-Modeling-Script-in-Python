#!/usr/bin/env python
# coding: utf-8

# # Linear Modeling in Python

# In[7]:

import pandas as pd #importing pandas
import matplotlib.pyplot as plt #importing mathplotlib
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

print("Linear Modeling in Python")
print(sys.argv[1])


# In[8]:


df = pd.read_csv('regrex1.csv') #reading file and saving as df 


# In[9]:


print(df.head()) #listing information as table 




# In[11]:


print(df['x'].head()) #listing x column


# In[12]:


print(df['y'].head()) #listing y column 



# In[14]:


plt.scatter(df['x'],df['y']) #creating scatter plot
plt.xlabel("x") #labeling x axis
plt.ylabel("y") #labeling y axis 
plt.show() # showing scatter plot
plt.savefig("Original Data.png")


# In[16]:


import numpy as np 
from sklearn.linear_model import LinearRegression
#importing np and Linear Regression


# In[17]:


x = np.array(df['x']).reshape(-1,1)
y = np.array(df['y']) #creating arrays for each axis 


# In[18]:


model = LinearRegression() #defining the model as a Linear Regression


# In[19]:


model.fit(x, y) #fitting the model with the data


# In[20]:


intercept = model.intercept_
slope = model.coef_
r_sq = model.score(x,y)
# calling for the intercept, slope, and r-squared of the model


# In[21]:

print("Summary data for Y-pred Model \n")
print(f"intercept: {intercept}")
print(f"slope: {slope}")
print(f"r squared: {r_sq}")
#print the intercept, slope, and r-squared of the model


# In[22]:


y_pred = model.predict(x) #using the model to predict the y values


# In[23]:


print(y_pred) #looking at the predictions 


# In[24]:


plt.plot(df['x'], y_pred) #plotting the linear regression prediction model
plt.xlabel("x") #adding x axis label
plt.ylabel("y") # adding y axis label
plt.show #showing the graph 
plt.savefig("Y Pred Model.png")

# In[25]:


plt.scatter(df['x'], df['y']) # plotting the orginal scatter plot data
plt.plot(df['x'], y_pred) # plotting the linear regression
plt.xlabel("x") #adding x axis label
plt.ylabel("y") # adding y axis label
plt.show
plt.savefig("Overlapping Data.png")


# In[ ]:




