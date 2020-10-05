#!/usr/bin/env python
# coding: utf-8

# # Import Library

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from keras.optimizers import Adam
from keras.utils import np_utils
from pyimagesearch.livenessnet import LivenessNet
import os
import time
import numpy as np
import joblib
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')


# # Loading Model

pickle_in = open("data/data_train.pickle","rb")
data_train = pickle.load(pickle_in)

pickle_in = open("data/labels_train.pickle","rb")
labels_train = pickle.load(pickle_in)	

pickle_in = open("data/data_test.pickle","rb")
data_test = pickle.load(pickle_in)

pickle_in = open("data/labels_test.pickle","rb")
labels_test = pickle.load(pickle_in)


# # Tunning KNN model

#List Hyperparameters that we want to tune.
#leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(n_neighbors=n_neighbors, p=p)
#Create new KNN object
model_neigh_tune = KNeighborsClassifier()
#Use GridSearch
model_neigh_tune = GridSearchCV(model_neigh_tune, hyperparameters, cv=5)
#Fit the model
start_time = time.time()
best_model_neigh = model_neigh_tune.fit(data_train, labels_train)
#Print The value of best Hyperparameters
print("Best: %f using %s" % (best_model_neigh.best_score_, best_model_neigh.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# # Tunning Logistic Regression

#List Hyperparameters that we want to tune.
dual=[True,False]
max_iter=[100,110,120,130,140]
C = [1.0,1.5,2.0,2.5]
hyperparameters = dict(dual=dual,max_iter=max_iter,C=C)
#Create new Logistic object
model_logistic_tune = LogisticRegression()
#Use GridSearch
model_logistic_tune = GridSearchCV(model_logistic_tune, hyperparameters, cv=5)
#Fit the model
start_time = time.time()
best_model_logistic = model_logistic_tune.fit(data_train, labels_train)
#Print The value of best Hyperparameters
print("Best: %f using %s" % (best_model_logistic.best_score_, best_model_logistic.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# # Tunning Random Forest

# Create the parameter grid based on the results of random search 
hyperparameters = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    #'min_samples_leaf': [3, 4, 5]
    #'min_samples_split': [8, 10, 12]
    #'n_estimators': [100, 200, 300, 1000]
}
# create a new random forest
model_rf_tune = RandomForestClassifier()
#Use GridSearch
model_rf_tune = GridSearchCV(model_rf_tune, hyperparameters, cv=5)
#Fit the model
start_time = time.time()
best_model_random_forest = model_rf_tune.fit(data_train, labels_train)
#Print The value of best Hyperparameters
print("Best: %f using %s" % (best_model_random_forest.best_score_, best_model_random_forest.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# # Tunning SVC Linear

# Create the parameter grid based on the results of random search 
hyperparameters = {
    'penalty': ["l1", "l2"],
    'loss': ["hinge","squared_hinge"],
    'C' : [0.1, 1, 10, 100, 1000]
}
# create a new random forest
model_svm_tune = LinearSVC()
#Use GridSearch
model_svm_tune = GridSearchCV(model_svm_tune, hyperparameters, cv=5)
#Fit the model
start_time = time.time()
best_model_svm = model_svm_tune.fit(data_train, labels_train)
#Print The value of best Hyperparameters
print("Best: %f using %s" % (best_model_svm.best_score_,best_model_svm.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

# # CNN

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 8
EPOCHS = 100
# initialize the optimizer and model
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model_cnn = LivenessNet.build(width=32, height=32, depth=3,
	classes=len(le.classes_))
model_cnn.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the network
start_time = time.time()
H = model_cnn.fit_generator(aug.flow(data_train, labels_train, batch_size=BS),
	validation_data=(data_test, labels_test), steps_per_epoch=len(data_train) // BS,
	epochs=EPOCHS)
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# # Prediction

prediction_rf = best_model_random_forest.predict(data_test)
prediction_svm = best_model_svm.predict(data_test)
prediction_knn = best_model_neigh.predict(data_test)
prediction_logistic = best_model_logistic.predict(data_test)
prediction_cnn = model_cnn.predict(data_test, batch_size=BS)

# # Accuracy Best Models

# accuracy
print("Modelo SVM",
classification_report(labels_test,prediction_svm))

print("Modelo Random Forest",
classification_report(labels_test,prediction_rf))

print("Modelo KNN",
classification_report(labels_test,prediction_knn))

print("Modelo Logistico",
classification_report(labels_test,prediction_logistic))

print("Modelo CNN",
classification_report(labels_test,prediction_cnn))


# # Save Models

# Random Forest
filename = 'models/best_model_random_forest.joblib'
joblib.dump(best_model_random_forest, filename)

# KNN
filename = 'models/best_model_knn.joblib'
joblib.dump(best_model_neigh, filename)

# SVM
filename = 'models/best_model_svm.joblib'
joblib.dump(best_model_svm, filename)

# Logistic
filename = 'models/best_model_logistic.joblib'
joblib.dump(best_model_logistic, filename)

# CNN
filename = 'models/model_cnn.joblib'
joblib.dump(model_cnn, filename)
