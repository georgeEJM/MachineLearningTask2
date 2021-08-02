#!/usr/bin/env python
# coding: utf-8

# In[64]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn import preprocessing
import random


# In[65]:


path = 'C:/Users/Owner/Documents/ML/clinData.xlsx' # Data path
data = pd.read_excel(path) # Reads in file
data = data.dropna(how='all') # Removes Na values

graph_info = data.copy() # Makes copy for later use
label = data['Status'] # Gets label (y)
data.pop('Status') # Removes label from data

X = np.array(data) # Creates X from data

label = label.replace(['healthy'], 0) # Binarises labels
label = label.replace(['cancerous'], 1)

y = np.array([label]) # Creates array from Y 
y = y.T # Transpose to match dimension

X = X / np.linalg.norm(X) # Normalise data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1) # Gets training/test split


# In[66]:


data.describe() # Summarises data (mean, min, etc)


# In[67]:


# Verifies shapes 
print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {y.shape}")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")


# In[7]:


sns.set_style("whitegrid") # Boxplot of age and status
sns.boxplot(x = 'Status', y = 'Age', data = graph_info).set_title('Boxplot of age values grouped by status')
plt.show()


# In[8]:


healthy = graph_info[graph_info.Status.isin(['healthy'])] # Uses graph info for readability (should plot this before modification)
cancerous = graph_info[graph_info.Status.isin(['cancerous'])]

ax = healthy[['BMI']].plot.kde() # Plots healthy BMI
cancerous[['BMI']].plot.kde(ax=ax) # Plots unhealthy BMI
ax.legend(['Healthy', 'Cancerous']) # legend
ax.set_title('Density plot of BMI values based on status') #Titles
ax.set_xlabel('BMI') # BMI
plt.show()


# In[32]:


def TrainModel(model, X_train, y_train, X_test, y_test, epochs): # Function to train model
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy']) # Compiles the model
    # Creates history of loss and accuracy as network is trained
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_data=(X_test, y_test)) 
    scores = model.predict(X_test, verbose=0) # Predictions for X test
    acc = model.evaluate(X_test, y_test) # Evaluates model effectiveness on validation data
    print("Accuracy: %.2f%%" % (acc[1]*100)) # Prints accuracy
    return history, scores # Returns history for plotting and y_pred


# In[33]:


ann=keras.models.Sequential() # Creates model
ann.add(keras.layers.Dense(500, activation='sigmoid')) # Adds 2 dense layers
ann.add(keras.layers.Dense(500, activation='sigmoid'))
ann.add(keras.layers.Dense(1, activation='sigmoid')) # Adds output layer


# In[34]:


tic = time.perf_counter() # Timer
history, y_pred = TrainModel(ann, X_train, y_train, X_test, y_test, 5000) # Trains network 
toc = time.perf_counter() # Timer
print(f"Training and testing took {toc - tic:0.4f} seconds") # How long training took


# In[14]:


y_pred_bool = np.argmax(y_pred, axis=1) # Boolean for correct vals


# In[62]:


print(classification_report(y_test, y_pred_bool)) # Prints precision, recall, etc using predicted against y_test.


# In[16]:


sns.distplot(history.history['accuracy'], kde = True) # Shows distribution of scores


# In[60]:


losschart = plt.figure(figsize = (5, 5)) # Plots loss

plt.plot(history.history['val_loss'], label="Training loss")
plt.plot(history.history['loss'], label="Testing loss")
plt.legend(prop={'size': 8})
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Plotting loss over epochs")
plt.show()


# In[61]:


accchart = plt.figure(figsize = (5, 5)) # Plots accuracy

plt.plot(history.history['val_accuracy'], label="Test accuracy")
plt.plot(history.history['accuracy'], label="Training accuracy")
plt.legend(prop={'size': 8})
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Plotting loss over epochs")
plt.show()


# In[68]:


def kfold(X, y, n_folds=0): # Function for k-folds
    
    fold_size = int(len(X) / n_folds) # Gets size of each fold
    if n_folds > int(len(X)): # If number of folds exceeds the maximum possible
        print(f"Number of folds exceeds maximum number of folds possible ({int(len(X))})") # Error message
        return np.empty(0), np.empty(0) # Returns empty arrays
    fold_data_x = X.copy() # Copies data for modification
    fold_data_y = y.copy()
    
    folds_X = [] # Declaration of empty arrays
    folds_y = []
    
    for _ in range(n_folds): # For every fold
        folds_X.append(fold_data_x[0:fold_size]) # Appends the first selection of data to relevant array
        fold_data_x = fold_data_x.drop(fold_data_x.index[0:fold_size]) # Removes this from copy of data
        
        # Repeats for y
        folds_y.append(fold_data_y[0:fold_size])
        fold_data_y = fold_data_y.drop(fold_data_y.index[0:fold_size])
        
    return folds_X, folds_y # Returns two folds


# In[69]:


folds_info = graph_info.reindex(np.random.permutation(graph_info.index)) # Shuffles copy of data
folds_info = folds_info.reset_index(drop=True) # Resets the index for permutation
# Goes through data pre-processing for shuffled data___
folds_label = folds_info['Status'] 
folds_info.pop('Status')

folds_label = folds_label.replace(['healthy'], 0)
folds_label = folds_label.replace(['cancerous'], 1)

folds_info = folds_info / np.linalg.norm(folds_info)
#______________________________________________________

n_folds = 10 # Number of folds
X, y = kfold(folds_info, folds_label, n_folds) # X and y folds


# In[73]:


# Creates ANN for folded data
ann_fold=keras.models.Sequential()
ann_fold.add(keras.layers.Dense(500, activation='sigmoid'))
ann_fold.add(keras.layers.Dense(500, activation='sigmoid'))
ann_fold.add(keras.layers.Dense(1, activation='sigmoid'))

avg_acc = 0 # Average accuracy

epochs = 1 # Number of epochs
ann_fold.compile(optimizer='adam', loss='mse', metrics=['accuracy']) # Compiles the network

why_X = pd.DataFrame(X[0]) # Creates dataframe from first list in X 
y_stretch = np.array(y[0]) # Creates array from first list in y

for i in range(1, n_folds): # For every fold not including those used for dataframe initialisation
    y_stretch = np.append(y_stretch, y[i]) # Adds y[i] to y
    why_X = pd.concat([why_X, X[i]]) # Concatenates the X dataframe and X[i]
    
why_y = pd.DataFrame(y_stretch) # Creates dataframe from y array

tic = time.perf_counter()

for i in range(n_folds): # For every fold
    cur_X = why_X.copy() # Creates copies for modification
    cur_y = why_y.copy()
    
    test_X = X[i] # Gets test values for X and Y
    test_y = y[i].astype(int)

    train_X = np.array(cur_X.drop(index=test_X.index)) # Drops test indices from dataframe copies to generate training values.
    train_y = np.array(cur_y.drop(index=test_y.index))
    train_y = train_y.squeeze().astype(int) # Formats y
    
    test_X = np.array(test_X) # Transforms test_X into array
    
    history = ann_fold.fit(train_X, train_y, epochs=epochs, verbose=0) # Fits current folds
    scores = ann_fold.evaluate(test_X, test_y, verbose=0) # Evaluates
    avg_acc += scores[1] * 100 # Adds to average acc
    print("Accuracy: %.2f%%" % (scores[1]*100)) # Prints accuracy
toc = time.perf_counter()
print(f"Training and testing took {toc - tic:0.4f} seconds") # Details time
print(f"Average accuracy: {avg_acc / n_folds}%") # Details average accuracy


# In[ ]:




