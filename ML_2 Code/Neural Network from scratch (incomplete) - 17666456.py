#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Non working neural network. Decreases cost to nearly 0 but cannot produce accurate results.
import numpy as np
from numpy.random import randn
import copy
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.special import expit, logit
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[2]:


np.random.seed(1234) # Random seed


# In[49]:


# Data pre-processing and gathering
path = 'C:/Users/Owner/Documents/ML/clinData.xlsx'
data = pd.read_excel(path)
data = data.dropna(how='all')

data = data.sample(frac = 1)

label = data['Status']
data.pop('Status')

test = data.to_numpy()
X = np.matrix(test)

label = label.replace(['healthy'], 0)
label = label.replace(['cancerous'], 1)

y = np.array([label])
y = y.T

X = X / np.linalg.norm(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1)


# In[35]:


def sigmoid (input): # Sigmoid 
    return expit(input)

def sigmoid_deriv(input): # Sigmoid derivative
    return np.multiply(sigmoid(input), (1 - sigmoid(input)))


# In[45]:


class HiddenLayer: # Class for hidden layer
    # Parameters on __init__
    def __init__(self, n_inputs, n_neurons):
        self.neurons = n_neurons
        self.weights = 2 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))
        
    def forward(self, input): # Forward propagation
        self.last_input = input
        output = (input @ self.weights) + self.bias
        output_sig = sigmoid(output)
        self.output = output_sig
        return output_sig
    
    # Backpropagation
    def backprop(self, dc_do_backprop, do_dnet_backprop, lr):
        do_dnet = np.array(sigmoid_deriv(self.last_input))
        dnet_dw = np.array(self.last_input)
        
        combine = do_dnet * dnet_dw
        
        dc_do = np.array(dc_do_backprop) * do_dnet_backprop

        weight_change = np.dot(combine.T, dc_do)
        self.weights -= weight_change * lr
        self.bias -= 1 * np.sum(dc_do_backprop, axis = 0) * lr
        return dc_do, do_dnet


# In[50]:


class OutputLayer: # Output layer
    # Parameters on __init__
    def __init__(self, n_inputs, n_neurons):
        self.weights = 2 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))
        
    def forward(self, input): # Forward propagation
        self.last_input = input
        output = (input @ self.weights) + self.bias
        output_sig = sigmoid(output)
        self.output = output
        self.output_sig = output_sig
        return output_sig
    
    def backprop(self, dc_do, lr): # Backpropagation (assumes this is the final layer)
    
        do_dnet = np.array(sigmoid_deriv(self.last_input))
        dnet_dw = np.array(self.last_input)
        
        combine = do_dnet * dnet_dw

        weight_change = np.dot(combine.T, dc_do)
        self.weights -= weight_change * lr
        self.bias -= 1 * np.sum(dc_do, axis = 0) * lr
        return do_dnet


# In[83]:


hidden_layer_1 = HiddenLayer(9, 500) # Creates layers
hidden_layer_2 = HiddenLayer(hidden_layer_1.neurons, 500)
output_layer = OutputLayer(hidden_layer_2.neurons, 1)

def train(x_train, y_train, epochs, lr): # (tries to) Trains network 
    
    total_correct = [] # Defaultvalues
    total_cost = []
    for i in range(epochs): # For every epoch
        
        n_correct = 0
        
        # Sends data through every output and normalises the output
        output_1 = hidden_layer_1.forward(x_train) 
        output_1 = output_1 / np.linalg.norm(output_1)
        output_2 = hidden_layer_2.forward(output_1)
        output_2 = output_2 / np.linalg.norm(output_2)
        final_output = output_layer.forward(output_1)
        

        for j in range(final_output.shape[0]): # Squashes data to value
            if (final_output[j]) <= 0.5:
                status = 0
            else:
                status = 1

            if status == y_train[j]: # Add to n_correct if prediction is correct
                n_correct += 1
        
        mse = 1/len(final_output) * (np.sum(y_train - final_output)**2) # Calculates mean squared error    
        total_correct.append([n_correct]) # Adds values to show history
        total_cost.append([mse])
        
        dcost_dao = final_output - y_train # First cost metric for backprop
        # Creates weight gradients and computes backprop for every layer
        output_do_dnet = output_layer.backprop(dcost_dao, lr) 
        hid2_dc_do, hid2_do_dnet = hidden_layer_2.backprop(dcost_dao, output_do_dnet, lr)
        hidden_layer_1.backprop(hid2_dc_do, hid2_do_dnet, lr)
    print(f"Epoch {i} - Number of correct predictions: {n_correct}") # Prints predictions etc.
    cost = 1/len(final_output) * (np.sum(y_train - final_output) ** 2)
    print(f"Cost of this outcome: {mse} \n")
    
    return total_correct, total_cost # History
    
def forward(x, y): 
    # Forward prop
    output_1 = hidden_layer_1.forward(x)
    output_2 = hidden_layer_2.forward(output_1)
    final_output = output_layer.forward(output_1)
    
    n_correct = 0
    for j in range(len(final_output)): # Calculates accuracy
        if (final_output[j]) < 0.5:
            status = 0
        else:
            status = 1
        if status == y_train[j]:
            n_correct += 1
        print(status)
    
    print(f"Number of correct predictions: {n_correct}, accuracy: {(n_correct / len(y)) * 100}%") # Prints predictions etc.
    cost = 1/len(final_output) * (np.sum(y - final_output) ** 2)
    print(f"Cost of this outcome: {cost} \n")
    return  final_output


# In[84]:


correct, cost = train(X_train, y_train, 100, 1e-2) # Trains network


# In[85]:


plt.plot(cost)# Cost graph


# In[86]:


plt.plot(correct) # Accuracy graph


# In[87]:


output = forward(X_test, y_test) # Y_pred


# In[ ]:




