# Neuro-Fuzzy-System
We try to implement a Neuro-Fuzzy model system where the goal is to create an rule based  classification to classify the data points based on limited features. We focused on classifying binary data, using this method. In this model, the data first goes to the Dynamic clustering system, which divides every feature into clusters. The output from dymanic clustering is fed into the fuzzification system that identifies a data feature with a cluster with the help of membership functions. The fuzzification module output is a binary list, which is fed into the neural network in the training phase to figure out the most essential feature. Once the top features are selected, a rule is created with those features to classify the data. We use the rule based classification for testing.

![Pipeline Diagram](images/pipeline.PNG)

# Implementation:

## Dynamic clustering

Dynamic clustering is a form of incremental machine learning. It is better form of clustering than k-means as it does not require predefined value of clusters. Dynamic clustering embraces many different scenarios like dynamic cluster & dynamic feature.

### Functioning of dynamic clustering:

Initially, the feature column will be sorted on the basis of values and then initial clusters will be formed on the basis of target class. In this process, while iterating the feature column if the data points follow same class then they will be put under the same cluster otherwise a new cluster will be formed. In the second stage, centroid for each cluster will be calculated. A threshold value was also defined for agglomeration of clusters. If the value of no. data points are less than the threshold in a cluster than this cluster will be merged to a cluster having closest value of centroid. Lastly, Mean and standard deviation for these clusters calculated and fed to the fuzzyfication part of algorithm.

![Dynamic clustering example](images/dynamic_clustering.PNG)



## Fuzzification
Fuzzification accepts inputs from dynamic clustering. Dynamic clustering
outputs the number of clusters for each feature and means and standard deviations
for each cluster. Fuzzification takes each datapoint and
pass through the membership functions which identifies the probability
of each datapoint being member of cluster based on mean and std
deviation of the clusters. It then creates a binary output that
assign the datapoint to the cluster with highest memebership value.

![Fuzzification Diagram](images/fuzzification.PNG)

## Neural-Network

Firstly we train the neuro network with inputs fed from Fuzzification. Using the Sigmoid activation function and the associated backwards propagation methods - including an error value from the differences in expected outputs and actual outputs.
Usually this results in clustering of values at either 0 or 1, but since we're solely interested in the trained weight values - this isn't a problem.
We then embed into each weight group a number to indicate the corresponding Fuzzified feature. Before sorting and then executing a rule extraction method, where we can later retreve the feature from said embedding.


## Rule based classification

We used the neural network to pick out best weights on the network, which inturn pointed out the most influencial clusters on the data set. Once we had those, we created a simple rule for classification based on the fuzzified output of the data set. 



# Results
We ran the algorithm on two different data sets, the Breast cancer data set and the Mesothelioma data set. Our results were satisfactory. 

Show below is the output of the Breast cancer data set. The Breast cancer data set had 31 featurs. The accuracy of the rules with the top 3 features is about 68.4%. 

```
The accuracy of our rules  with top 3 features is: 0.6842105263157895
The accuracy of our rules  with top 5 features is: 0.6578947368421053
The accuracy of our rules  with top 10 features is: 0.6052631578947368

```




This is the accuracy for the Mesothelioma data set, in which we had to classify if the patient had Mesothelioma or not. The Mesothelioma data set had 34 features. The accuracy of rule based classification with top 3 features is 80%.


```
The accuracy of our rules  with top 3 features is: 0.8
The accuracy of our rules  with top 5 features is: 0.7692307692307693
The accuracy of our rules  with top 10 features is: 0.5692307692307692

```
# References
1. Phichit Napook and Narissara Eiamkanitcaht. 2015. Forecasting algorithm developed from   the neuro: fuzzy system. In Adjunct Proceedings of the 2015 ACM International Joint Conference on Pervasive and Ubiquitous Computing and Proceedings of the 2015 ACM International Symposium on Wearable Computers(UbiComp/ISWC'15 Adjunct). Association for Computing Machinery, New York, NY, USA, 1189–1196. DOI:https://doi.org/10.1145/2800835.2800983

+ N. Eiamkanitchat, N. Theera-Umpon and S. Auephanwiriyakul, "A novel neuro-fuzzy method for linguistic feature selection and rule-based classification," 2010 The 2nd International Conference on Computer and Automation Engineering (ICCAE), 2010, pp. 247-252, doi: 10.1109/ICCAE.2010.5451487.

+ P. Wongchomphu and N. Eiamkanitchat, "Enhance Neuro-fuzzy system for classification using dynamic clustering," The 4th Joint International Conference on Information and Communication Technology, Electronic and Electrical Engineering (JICTEE), 2014, pp. 1-6, doi: 10.1109/JICTEE.2014.6804071.

+ Heisnam Rohen Singh, Saroj Kr. Biswas, Biswajit Purkayastha, A neuro-fuzzy classification technique using dynamic clustering and GSS rule generation, Journal of Computational and Applied Mathematics, Volume 309, 2017, Pages 683-694, ISSN 0377-0427, https://doi.org/10.1016/j.cam.2016.04.023.



