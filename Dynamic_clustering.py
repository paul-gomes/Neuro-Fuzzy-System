#!/usr/bin/env python
# coding: utf-8

# In[2]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('C:/Users/Hp 840/Downloads/bezdekIris.data',names = ['a','b','c','d','class'])

def clustering(df,count):
    df.sort_values(by=df.columns[count],inplace=True)
    df.reset_index(drop=True, inplace=True)
    Target = list(df['class'])
    q = df['class'][0]
    Cluster_form = list()
    cl = list()
    for i in range(0,len(Target)):
        if Target[i] == q:
            cl.append(df[df.columns[count]][i])
        else:
            q = df['class'][i]
            Cluster_form.append(cl)
            cl = list()
            cl.append(df[df.columns[count]][i])
    Cluster_form.append(cl)
#clustering of feature values
    #print("CLuster",z)
    s1 = 0
    for v in Cluster_form: 
        s1+=len(v)
    Threshhold = s1/len(Cluster_form) 
    #print("s1",s1)
    #print(len(z))
    #print("t",Threshhold)
    return centroid(Cluster_form,Threshhold)


def centroid(Cluster_form,Threshhold):
# list as a input for centroid function
    def cen(i):
        for i in Cluster_form:
            ci = sum(i)/len(i)
            return ci
    
    #print("len",len(z))

#cluster are combine with other cluster based on less distance between centroid of clusters
    for i in Cluster_form:
        if len(i) < Threshhold:
            if (cen(i)-cen(Cluster_form[Cluster_form.index(i)-1]))<(cen(i)-cen(Cluster_form[Cluster_form.index(i)+1])): 
                for t in i:
                    Cluster_form[Cluster_form.index(i)-1].append(t)     
            elif (cen(i)-cen(Cluster_form[Cluster_form.index(i)-1]))>(cen(i)-cen(Cluster_form[Cluster_form.index(i)+1])):
                for t in i:
                    Cluster_form[Cluster_form.index(i)+1].append(t)

#clusters which have member in cluster less than Fth are eliminated
    new = list()
    for i in Cluster_form:
        if len(i) > Threshhold:
            new.append(i)
    #print("new",len(new))
    #print("newnn",new) #list with newly formed cluster after elimination

#mean is calculated
    mean = list()
    for i in new:
        mean.append(sum(i)/len(i))  
    #print(mean)
    #print('\n')

#sd is calculated
    SD = list()
    for i in new:
        SD.append(np.std(i)) 
    #print(SD)
    return mean,SD


def Dynamic_clustering():
    count=0   
    MEAN=[]
    Std_deviation=[]
    while count<len(df.columns)-1:
    
        a,b=clustering(df,count)
        MEAN.append(a)
        Std_deviation.append(b)
    
    #print(len(df.columns))
        count+=1
    #print(MEAN)
    #print(len(MEAN))
    #print(Std_deviation)
    return MEAN,Std_deviation

M,S=Dynamic_clustering()
print(M)

print(S)
    

        


# In[ ]:




