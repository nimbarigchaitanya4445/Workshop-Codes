#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


data = pd.read_csv('mnist_train.csv');


# In[28]:


data.head()


# In[29]:


a = data.iloc[3,1:].values


# In[31]:


a = a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[33]:


df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]


# In[34]:


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state=4)


# In[35]:


x_train.head()


# In[36]:


y_train.head()


# In[38]:


rf = RandomForestClassifier(n_estimators=100)


# In[40]:


rf.fit(x_train, y_train)


# In[41]:


pred = rf.predict(x_test)


# In[45]:


print(pred)


# In[43]:


s = y_test.values
count = 0
for i in range(len(pred)):
    if pred[i] == s[i]:
        count = count + 1;


# In[46]:


print(count)


# In[47]:


len(pred)


# In[49]:


accuracy = (count/len(pred))*100
print(accuracy)

