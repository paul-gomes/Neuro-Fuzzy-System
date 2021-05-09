import pandas as pd
import numpy as np
from Dynamic_Clustering import *
from Fuzzification import *
from NeuralWeightsFunction import *

bcw_data = pd.read_csv("Data/breast_cancer_wisconsin.csv", header = 0)
bcw_data.drop('id', axis=1, inplace=True) #dropping the id col

#selects the features you need
data = bcw_data[bcw_data.columns[1:4]]

#targets classes 
t= bcw_data[['diagnosis']].replace(['M','B'],[0,1])

#this is dataframe with cleaned data which has the features and the class(either 0(M) or 1(B))
df = pd.concat([data, t], axis=1)
df = df.rename(columns={"radius_mean": "a","texture_mean":"b","perimeter_mean":"c","diagnosis":"class"})
print(df.sample(n=5))

#means and sds from dynamic clustering
MEAN,Std_deviation = Dynamic_clustering(df)

desired_t = [[1,0],[0,1]]
#targets in list format for neural netwrork eg. [0,1] for class being B [1,0] for class being M
targets = np.array([desired_t[int(x)] for x in df.values[:,3:4]])
print(targets[:5])

#vlaues after fuzzification.. it's a list of list.
fuzzy_values = []
for index, row in data.iterrows():
    f = Fuzzification(row, MEAN, Std_deviation)
    fuzzy_val = f.fuzzify()
    fuzzy_values.append(sum(fuzzy_val, []))
    
print(fuzzy_values)


listofnewweightsbiases = sigmoid_training_special(fuzzy_values, targets)
classifier_x = calc_x(fuzzy_values, listofnewweightsbiases[0])

print(listofnewweightsbiases)
print(classifier_x)