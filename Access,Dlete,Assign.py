#!/usr/bin/env python
# coding: utf-8

# In[28]:


dictionary = {"1":2,"3":4,"5":6,"7":8}
tup = ('Derivative','Python','Software',1,2)
def elementassign():
    n = int(input("Enter the number of elements to add "));
    lst = [];
    for n in range(0,n):
        x = int(input())
        lst.append(x);
    print(lst);
def accessingelements(pos):
    tup = ('Derivative','Python','Software',1,2)
    print(tup[pos])
def deleteelement(key):
    dictionary = {"1":2,"3":4,"5":6,"7":8}
    if key in dictionary:
        del dictionary[key];
    print(dictionary);
i = int(input("1. Assign Element to a tuple\n2. Access element from a tuple\n3. Delete element from dictionary\n"));
if(i == 1):
    elementassign();
if(i == 2):
    pos = int(input("Enter the position to be accessed "));
    accessingelements(pos);
if(i == 3):
    key = input("Enter the key to be deleted ");
    deleteelement(key);


# In[ ]:





# In[ ]:




