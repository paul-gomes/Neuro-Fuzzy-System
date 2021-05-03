#!/usr/bin/env python3

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter

#plt.close("all")

#------------------------------------------------------------------------------------------------------
# Forecasting Algorith Developed from the Neuro - Fuzzy System
# Authors: Phichit Napook and Narissara Eiamkanitcaht
#       Chiang Mai University,Thailand
# Copyright 2015 © 2015 ACM 978-1-4503-3575-1/15/09 $15.00.
# http://dx.doi.org/10.1145/2800835.2800983
#------------------------------------------------------------------------------------------------------

# Testing Comment for Compilation
print('Starting')

#------------------------------------------------------------------------------------------------------
#    Neural Network
#------------------------------------------------------------------------------------------------------

weights  = np.random.rand(2,3)
inputs = [[3,3,3],[-3,-3,-3],[4,4,-4]]
biases = np.random.rand(2)
expectedOutput = [1,0,1]

def my_neuron(weightList, inputList, bias, my_function):
	dotproduct = np.dot(weightList,inputList)
	fullproduct = dotproduct + bias
	return my_function(fullproduct)

def my_sigmoid(value):
	value2 = value * -1
	sigmoid = 1 / (1 + np.exp(value2))
	return sigmoid

def my_derivative_sigmoid(value):
	sigmoid = my_sigmoid(value) * (1 - my_sigmoid(value))
	return sigmoid

def sigmoid_neuron(weightList, inputList, bias):
	return my_neuron(weightList, inputList, bias, my_sigmoid)

def singleInput_neuron_layer(EveryNeurons_weightList, inputList, EveryNeurons_bias, Individual_neuron):
	full_output = []
	for weightList,bias in zip(EveryNeurons_weightList, EveryNeurons_bias):
		single_output = Individual_neuron(weightList, inputList, bias)
		full_output.append(single_output)
	return full_output

def neuron_layer(EveryNeurons_weightList, Every_inputList, EveryNeurons_bias, Individual_neuron):
	full_output = []
	for inputList in Every_inputList:
		single_output = singleInput_neuron_layer(EveryNeurons_weightList, inputList, EveryNeurons_bias, Individual_neuron)
		full_output.append(single_output)
	return full_output

def simple_error(expectedValue, actualValue):
	diff = (expectedValue - actualValue) * (expectedValue - actualValue)
	return diff

def layer1_derivative_chain(weightList, expectedValue, actualValue, Adenduminputs, maybeBias):
	dotproduct = np.dot(weightList,Adenduminputs)
	fullproduct = dotproduct + maybeBias
	full_output = []
	sigmoid_derivative = my_derivative_sigmoid(fullproduct)
	#for w_weight in weightList:
	for in_input in Adenduminputs:
		per_weight_derivative = sigmoid_derivative * in_input #w_weight
		full_output.append(per_weight_derivative)
	return full_output

def layer1_bias_chain(weightList, expectedValue, actualValue, Adenduminputs, maybeBias):
	dotproduct = np.dot(weightList,Adenduminputs)
	fullproduct = dotproduct + maybeBias # simple_error(expectedValue, actualValue)
	return my_derivative_sigmoid(fullproduct)

def layer1_weight_adjustment(weightList, bias, expectedValue, actualValue, Adenduminputs):
	weight_adjustments = layer1_derivative_chain(weightList, expectedValue, actualValue, Adenduminputs, bias)
	bias_adjustment = layer1_bias_chain(weightList, expectedValue, actualValue, Adenduminputs, bias)
	new_weightList = np.add(weightList, weight_adjustments) #np.add for opposite affect subtract
	new_bias = bias + bias_adjustment
	return [new_weightList, new_bias]

def training_attempt(weightList, bias, inputs, sigmoid_neuron, expectedValue):
	actualValue = singleInput_neuron_layer(weightList,inputs,bias,sigmoid_neuron)
	return layer1_weight_adjustment(weightList, bias, expectedValue, actualValue, inputs)

def multiple_training_attempts(weightLists, biases, inputLists, sigmoid_neuron, expectedValues):
	new_weightLists = []
	new_biasList = []
	actualValues = neuron_layer(weightLists,inputLists,biases,sigmoid_neuron)
	for weightList,bias in zip(weightLists, biases):
		for inputs, expectedValue, actualValue in zip(inputLists, expectedValues, actualValues):
			adjustments = layer1_weight_adjustment(weightList, bias, expectedValue, actualValue[0], inputs)
			weightList = adjustments[0]
			bias = adjustments[1]
		new_biasList.append(bias)
		new_weightLists.append(weightList)
	return [new_weightLists, new_biasList]

def bulk_training(weightLists, biases, inputLists, sigmoid_neuron, expectedValues):
	for x in range(1):
		listofnewweightsbiases = multiple_training_attempts(weightLists, biases, inputLists, sigmoid_neuron, expectedValues)
		weightLists = listofnewweightsbiases[0]
		biases = listofnewweightsbiases[1]
	return listofnewweightsbiases


def neruotest():
	print("Starting Weights and Answer Estimation")
	print([weights, biases])
	print(neuron_layer(weights,inputs,biases,sigmoid_neuron))
	listofnewweightsbiases = bulk_training(weights,biases, inputs, sigmoid_neuron, expectedOutput)
	print("Ending Weights and Answer Estimation")
	print(listofnewweightsbiases)
	print(neuron_layer(listofnewweightsbiases[0],inputs,listofnewweightsbiases[1],sigmoid_neuron))

#neruotest()

#------------------------------------------------------------------------------------------------------
#    Quadratic Function
#------------------------------------------------------------------------------------------------------

training_weights = weights

#Based on idea that the paper talks about the length of the weights.
maxlength = len(training_weights[0]) # As in the total number of weights that are in a singular node

x_0 = 0 #Lowest weight is 0 in sigmoid
x_1 = random.randint(1,maxlength)#np.random.choice(training_weights)#np.random.rand(1)[0] #Random weight in sigmoid between 0 and 1
x_2 = maxlength#max(training_weights) #Highest weight is 1 in sigmoid
x_3 = 0 # Not Initialized Yet.

# A, if a weight set applied to an input set is equal to 1, else 0

def calc_x_uninitialized(x,training_data,training_weights):
	calc_sum = 0
	for training_weight in training_weights:
		#Big_A = 1
		for training_item in training_data:
			Big_A = 1
			j = 0
			while ((Big_A == 1) and (j < x) and (j < len(training_weights[0]))) :
				if (training_item[j] * training_weight[j] >= .9):
					Big_A = 1
				else:
					Big_A = 0
				j += 1
			calc_sum += Big_A
	return calc_sum

def calc_x3_internal(x_0,x_1,x_2, training_data, training_weights):
	def f(x):
		return calc_x_uninitialized(x,training_data,training_weights)
	top_half = f(x_2)*((x_0**2)-(x_1**2)) + f(x_0)*((x_1**2)-(x_2**2)) + f(x_1)*((x_2**2)-(x_0**2))
	bottom_half = 2*f(x_2)*(x_0-x_1) +  2*f(x_0)*(x_1-x_2) +  2*f(x_1)*(x_2-x_0)
	if bottom_half == 0:
		return top_half/.001
	return top_half/bottom_half

def calc_x(x_0,x_1,x_2, training_data, training_weights):
	def f(x):
		return calc_x_uninitialized(x,training_data,training_weights)
	x_3 = calc_x3_internal(x_0,x_1,x_2, training_data, training_weights)
	f_x_3 = f(x_3)
	f_x_2 = f(x_2)
	while ((f(x_0)- f(x_1))**2 >= 0.1):
		#do all this
		if (f_x_3 == f_x_2):
			x_2 = x_3
			f_x_2 = f_x_3

		#Iterate_random process of X_1 ???
		elif (f_x_3 < f_x_2):
			x_0 = x_3
			f_x_2 = f_x_3
		print("I'm here")

		x_1 = x_3
		f_x_1 = f_x_3
		top_half = f_x_2*((x_0**2)-(x_1**2)) + f(x_0)*((x_1**2)-(x_2**2)) + f_x_1*((x_2**2)-(x_0**2))
		bottom_half = 2*f_x_2*(x_0-x_1) +  2*f(x_0)*(x_1-x_2) +  2*f_x_1*(x_2-x_0)
		if bottom_half == 0:
			x_3 = top_half/.001
		else:
			x_3 = top_half/bottom_half
		f_x_3 = f(x_3)
	print(f_x_3)
	return x_1
#print(calc_x(x_0,x_1,x_2, inputs, weights))
# Use X_1 to creat the binary clasification somehow???

#classifier(x_0,x_1,x_2, new_inputs,  ,trained_weights)


def fulltest():
	print("Starting Weights and Answer Estimation")
	print([weights, biases])
	#print(neuron_layer(weights,inputs,biases,sigmoid_neuron))
	listofnewweightsbiases = bulk_training(weights,biases, inputs, sigmoid_neuron, expectedOutput)
	print("Ending Weights and Answer Estimation")
	print(listofnewweightsbiases)
	
	classifier_x = calc_x(x_0,x_1,x_2, inputs, listofnewweightsbiases[0])
	print("Classifier Rule Estimation")
	print(classifier_x)

	somevalue = 3
	def my_rule(x):
		if my_sigmoid(x) > classifier_x:
			return 1
		else:
			return 0
	print(my_rule(somevalue))


fulltest()

# Example Output
# Starting
# Starting Weights and Answer Estimation
# [array([[0.49534594, 0.19905671, 0.6490372 ],
#        [0.13796504, 0.03065438, 0.23359039]]), array([0.17716456, 0.01078745])]
# Ending Weights and Answer Estimation
# [[array([ 1.45900802,  1.1627188 , -0.31284341]), array([1.0748309 , 0.96752024, 0.33563812])], [0.44646131039396936, 0.2944449358994945]]
# I'm here
# 6
# Classifier Rule Estimation
# -0.5
# [Finished in 0.7s]