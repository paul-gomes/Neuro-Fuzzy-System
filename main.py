import pandas as pd
import numpy as np
from Dynamic_Clustering import *
from Fuzzification import *
from NeuralWeightsFunction import *
from sklearn.model_selection import train_test_split

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

train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

desired_t = [[1,0],[0,1]]
#targets in list format for neural netwrork eg. [0,1] for class being B [1,0] for class being M
train_t = np.array([desired_t[int(x)] for x in train.values[:,3:4]])
print(train_t[:5])

test_t = np.array([desired_t[int(x)] for x in test.values[:,3:4]])
print(test_t[:5])


#means and sds from dynamic clustering
MEAN,Std_deviation = Dynamic_clustering(train)

#vlaues after fuzzification.. it's a list of list.
fuzzy_values = []
train_f = train.iloc[: , :-1]

for index, row in train_f.iterrows():
    f = Fuzzification(row, MEAN, Std_deviation)
    fuzzy_val = f.fuzzify()
    fuzzy_values.append(sum(fuzzy_val, []))
    
print(fuzzy_values)


listofnewweightsbiases = sigmoid_training_special(fuzzy_values, train_t)
classifier_x = calc_x(fuzzy_values, listofnewweightsbiases[0])
alternative_top5_features = important_feature_selection(listofnewweightsbiases[0])



print(listofnewweightsbiases[0])
print(classifier_x)
print(len(listofnewweightsbiases[0][0]))
print(alternative_top5_features)