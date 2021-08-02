#!/usr/bin/env python
# coding: utf-8

# In[30]:


# Importing libraries
import numpy as np
import pandas as pd
import time
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


def entropy(y): # Calculates entropy
    hist = np.bincount(y) # Obtains frequency of elements
    ps = hist / len(y) 
    return -np.sum([p * np.log2(p) for p in ps if p > 0]) # Calculates entropy


# In[3]:


class Node: # Node for use in decision tree
    # Sets parameters on __init__
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf_node(self): # Returns a value if self.value has a value
        return self.value is not None


# In[4]:


def most_common_label(y): # Finds most common label for val
    counter = Counter(y) # Counts number of elements
    most_common = counter.most_common(1)[0][0] # Gets most common
    return most_common # Returns

class DecisionTree: # Decision tree
    # Sets paramaters on __init__
    def __init__(self, min_samples_split=5, max_depth=10, n_feats=None):
        self.min_samples_split = min_samples_split # The minimum samples required for a split
        self.max_depth = max_depth # How deep the tree will go
        self.n_feats = n_feats # Number of features
        self.root = None # Path of tree nodes
        
    def fit(self, X, y): # Fits features for X
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1]) # Finds minimum of features in X
        # Or self.n_feats, whichever is smaller
        self.root = self._grow_tree(X, y) # Grows the tree
    
    def _grow_tree(self, X, y, depth=0): # Function to grow tree, starts at depth 0
        n_samples, n_features = X.shape # Gets shape of X to identify number of features and samples
        n_labels = len(np.unique(y)) # Gets labels (classes)

        # Stopping Criteria
        if (depth >= self.max_depth 
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = most_common_label(y)
            return Node(value=leaf_value)
        # Gets features indices for random selection of data
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False) 

        # Greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        
        # Grows the nodes resulting from the split (child nodes)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)
    
    def _best_criteria(self, X, y, feat_idxs): # Selects best criteria
        best_gain = -1 # Default value is worse than any gain to stop incorrect learning
        split_idx, split_thresh = None, None # Default
        for feat_idx in feat_idxs: # For every feature index
            X_column = X[:, feat_idx] # Gets column
            thresholds = np.unique(X_column) # Gets threshold
            for threshold in thresholds: # For every threshold
                gain = self._information_gain(y, X_column, threshold) # Calculates individual info gain
                
                if gain > best_gain: #If gain is superior
                    best_gain = gain # Sets new values
                    split_idx = feat_idx
                    split_thresh = threshold
                    
        return split_idx, split_thresh
    
    def _information_gain(self, y, X_column, s_thresh): # Calculates greatest info gain
        parent_entropy = entropy(y) # Calculates entropy

        left_idxs, right_idxs = self._split(X_column, s_thresh) # S
        
        if len(left_idxs) == 0 or len(right_idxs) == 0: # If no valid split
            return 0

        n = len(y) # Length of y
        n_left, n_right = len(left_idxs), len(right_idxs) # Gets length of left and right
        entropy_left, entropy_right = entropy(y[left_idxs]), entropy(y[right_idxs]) # Calculates entropy of both
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right # Calculates child entropy

        info_gain = parent_entropy - child_entropy # Establishes overall info gain
        return info_gain
    
    def _split(self, X_column, s_thresh): # Split decision for node
        left_idxs = np.argwhere(X_column <= s_thresh).flatten() # Creates index for splits
        right_idxs = np.argwhere(X_column > s_thresh).flatten() 
        return left_idxs, right_idxs
    
    def predict(self, X): # Traverses tree and calculates predictions for every value of X
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node): # Traversal of tree
        if node.is_leaf_node(): # If at a leaf node, return it's value
            return node.value
        elif x[node.feature] <= node.threshold: # If not, continue to traverse left to right based on threshold
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


# In[25]:


def bootstrap_sample(X, y): # Gets smaller sample from data 
    n_samples = X.shape[0]
    # Random choice between 0 and n_samples
    idxs = np.random.choice(n_samples, size = n_samples, replace=True) 
    return X[idxs], y[idxs]

class RandomForest: # Random forest for decision tree
    # Sets paramaters on __init__
    def __init__(self, n_trees=5, min_samples_split=50, max_depth=5, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []
        
    def fit(self, X, y): # Fits data by creating multiple decision trees and fitting data to each one
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth = self.max_depth,
                               n_feats = self.n_feats) # Makes new tree with parameters
            X_sample, y_sample = bootstrap_sample(X, y) # Gets sample to send through tree
            tree.fit(X_sample, y_sample)
            self.trees.append(tree) # Adds tree to dictionary
            
    def predict(self, X): # Predicts by going through each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        # Returns most common label for every tree's prediction
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds] 
        return np.array(y_pred)
    
def accuracy(y_true, y_pred): # Calculates accuracy of results (%)
    accuracy = np.sum(y_true == y_pred) / len(y_true) * 100
    return accuracy
        


# In[17]:


path = 'C:/Users/Owner/Documents/ML/clinData.xlsx' # Data path
data = pd.read_excel(path) # Reads in file
data = data.dropna(how='all') # Removes Na values

k_data = data.copy() # Makes copy for later use
label = data['Status'] # Gets label (y)
data.pop('Status') # Removes label from data

X = np.array(data) # Creates X from data

label = label.replace(['healthy'], 0) # Binarises labels
label = label.replace(['cancerous'], 1)

y = np.array([label]) # Creates array from Y 
y = y.T # Transpose to match dimension
y = y.squeeze() # Gets labels to match conditions for tree
X = X / np.linalg.norm(X) # Normalise data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1) # Gets training/test split


# In[34]:


tic = time.perf_counter() # Timer
clf = RandomForest(min_samples_split=5, n_trees=100) # Creates random forest
clf.fit(X_train, y_train) # Fits data
y_pred = clf.predict(X_test) # Gets predictions

toc = time.perf_counter() # Timer
acc = accuracy(y_test, y_pred) # Calculates accuracy
              
print("Accuracy:", acc) # Prints results
print(f"Training and testing took {toc - tic:0.4f} seconds")


# In[33]:


print(classification_report(y_test, y_pred)) # Prints precision, recall, etc using predicted against y_test.


# In[26]:


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


# In[27]:


folds_info = graph_info.reindex(np.random.permutation(k_data.index)) # Shuffles copy of data
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


# In[28]:


clf_folded = RandomForest(n_trees=20) # Makes forest for fold data

why_X = pd.DataFrame(X[0]) # Creates dataframe from first list in X 
y_stretch = np.array(y[0]) # Creates array from first list in y

for i in range(1, n_folds): # For every fold not including those used for dataframe initialisation
    y_stretch = np.append(y_stretch, y[i]) # Adds y[i] to y
    why_X = pd.concat([why_X, X[i]]) # Concatenates the X dataframe and X[i]
    
why_y = pd.DataFrame(y_stretch) # Creates dataframe from y array

tic = time.perf_counter()
avg_acc = 0
for i in range(n_folds): # For every fold
    cur_X = why_X.copy() # Creates copies for modification
    cur_y = why_y.copy()
    
    test_X = X[i] # Gets test values for X and Y
    test_y = y[i].astype(int)

    train_X = np.array(cur_X.drop(index=test_X.index)) # Drops test indices from dataframe copies to generate training values.
    train_y = np.array(cur_y.drop(index=test_y.index))
    train_y = train_y.squeeze().astype(int) # Formats y
    
    test_X = np.array(test_X) # Transforms test_X into array

    clf_folded.fit(train_X, train_y) # Trains forest
    
    pred = clf_folded.predict(test_X) # Gets predictions
    acc = accuracy(test_y, pred) # Calculates accuracy
    avg_acc += acc
    print(f"Accuracy: {acc}%") # Prints accuracy
toc = time.perf_counter()
print(f"Training and testing took {toc - tic:0.4f} seconds")
print(f"Average accuracy: {avg_acc / n_folds}%") # Details average accuracy


# In[ ]:




