#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:37:33 2019

@author: yogeshverma
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold

df = pd.read_json("train.json)
df_test = pd.read_json("test.json")

# Training set data extraction
dataset_size = df.iloc[:,1].size
ingredients = df.iloc[:,2]
label = df.iloc[:, 0]

# Test set data extraction
df_test_size = df_test.shape[0]
test_ingredients = df_test.iloc[:, 1]
test_ingredients_list = [j for j in test_ingredients ]

#unique cuisines
unique_cuisines = df.iloc[: , 0].unique()

#ingredients list from training set
ingredients_list = [i for i in ingredients ]

#ingredients list from test set
test_ingredients_list = [j for j in test_ingredients ]

#unique ingredients numpy array from training set
unique_ingredients = (np.unique([item for sublist in ingredients_list for item in sublist]))
#unique ingredients numpy array from test set
unique_test_ingredients = (np.unique([item for sublist in test_ingredients_list for item in sublist]))

#unique ingredients list from training set
unique_ingredients_list = unique_ingredients.tolist()

#unique ingredients list from test set
unique_test_ingredients_list = unique_test_ingredients.tolist()

#total_unique_ingredients = np.unique(unique_ingredients_list + unique_test_ingredients_list).tolist()

#ingredients feature vector of N * d shape, where N is total number of rows of cuisines, and d is unique ingredients list
ingredients_fv = np.zeros((dataset_size, len(unique_ingredients_list)))

#Replace ingredients which are present as 1
for idx, dish_list in enumerate(ingredients_list):
    for dish in dish_list:
        if(dish in unique_ingredients_list):
            ingredients_fv[idx][unique_ingredients_list.index(dish)] = 1
    
#Gaussian Distribution Model
gaussian_model = GaussianNB()

#Bernoulli Distribution Model
bernoulli_model = MultinomialNB()

#K-Fold split of the training data using 
k = KFold(n_splits=3)
for train_index, test_index in k.split(ingredients_fv):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = ingredients_fv[train_index], ingredients_fv[test_index]
    y_train, y_test = label[train_index], label[test_index]
    gaussian_model.fit(X_train,y_train)
    gaussian_predict = gaussian_model.predict(X_test)
    

features_train =  ingredients_fv[ :26516, :]
features_test =  ingredients_fv[26516:39774, :]

labels_train = label[ : 26516]
labels_test = label[26516:39774]

gaussian_model.fit(features_train,labels_train)
gaussian_predict = gaussian_model.predict(features_test)

accuracy_gaussian = accuracy_score(labels_test, gaussian_predict)

bernoulli_model.fit(features_train,labels_train)
bernoulli_predict = bernoulli_model.predict(features_test)

accuracy_bern = accuracy_score(labels_test, bernoulli_predict)

log_model = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial').fit(features_train, labels_train)
log_predictions = log_model.predict(features_test)
accuracy_log = accuracy_score(labels_test, log_predictions)

log_model_train = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial').fit(ingredients_fv, label)


ingredients_fv_test = np.zeros((df_test_size, len(total_unique_ingredients)))

for idx, dish_list in enumerate(test_ingredients_list):
    for dish in dish_list:
        if(dish in total_unique_ingredients):
            ingredients_fv_test[idx][total_unique_ingredients.index(dish)] = 1
            
log_predictions_test = log_model_train.predict(ingredients_fv_test)
test_ids = df_test.iloc[:, 0]
predictions_df = pd.DataFrame(log_predictions_test)
predictions_df['id'] = test_ids
predictions_df.to_csv("predictions_test.csv")