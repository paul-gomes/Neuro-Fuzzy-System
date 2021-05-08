import pandas as pd
import numpy as np

bcw_data = pd.read_csv("Data/breast_cancer_wisconsin.csv", header = 0)
bcw_data.drop('id', axis=1, inplace=True) #dropping the id col

#selects the features you need
data = bcw_data[bcw_data.columns[1:4]]

#targets classes 
t= bcw_data[['diagnosis']].replace(['M','B'],[0,1])

df = pd.concat([data, t], axis=1)
print(df.sample(n=5))

desired_t = [[1,0],[0,1]]

#targets in list format
targets = np.array([desired_t[int(x)] for x in df.values[:,3:4]])
print(targets[:5])