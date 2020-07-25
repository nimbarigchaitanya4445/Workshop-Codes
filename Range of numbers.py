#!/usr/bin/env python
# coding: utf-8

# In[33]:


lst = []
len = input("Enter the number terms in the list ");
len = int(len);
print("Enter the elements of the list ");
for i in range(0,len) :
    ele = int(input());
    lst.append(ele);
print(lst);
ll = int(input("Enter the lower limit "));
hl = int(input("Enter the higher limit "));
for x in lst:
    if(x>=ll):
        if(x<=hl):
            print(x);
            


# In[ ]:





# In[ ]:




