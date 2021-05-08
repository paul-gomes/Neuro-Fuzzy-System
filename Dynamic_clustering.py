#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

xy= pd.read_csv('C:/Users/Hp 840/Downloads/bezdekIris.data',names = ['a','b','c','d','class'])
xy.sort_values(by=['a'],inplace= True)
xy.reset_index(drop=True, inplace=True)
L = list(xy['class'])
q = xy['class'][0]
z = list()
cl = list()
for i in range(0,len(L)):
    if L[i] == q:
        cl.append(xy['a'][i])
    else:
        q = xy['class'][i]
        z.append(cl)
        cl = list()
        cl.append(xy['a'][i])
z.append(cl)
#clustering of feature values
print(z)
s1 = 0
for i in z: 
    s1+=len(i)
Threshhold = s1/len(z) 
print(Threshhold)



# list as a input for centroid function
def cen(i):
    for i in z:
        ci = sum(i)/len(i)
        return ci
    
print(len(z))

#cluster are combine with other cluster based on less distance between centroid of clusters
for i in z:
    if len(i) < Threshhold:
        if (cen(i)-cen(z[z.index(i)-1]))<(cen(i)-cen(z[z.index(i)+1])): 
            for t in i:
                z[z.index(i)-1].append(t)     
        elif (cen(i)-cen(z[z.index(i)-1]))>(cen(i)-cen(z[z.index(i)+1])):
              for t in i:
                z[z.index(i)+1].append(t)

#clusters which have member in cluster less than Fth are eliminated
new = list()
for i in z:
    if len(i) > Threshhold:
        new.append(i)
print(len(new))
print(new) #list with newly formed cluster after elimination

#mean is calculated
mean = list()
for i in new:
        mean.append([sum(i)/len(i)])  
print(mean)
print('\n')

#sd is calculated
SD = list()
for i in new:
    SD.append([np.std(i)]) 
print(SD)
            
        


# In[ ]:





# In[ ]:




