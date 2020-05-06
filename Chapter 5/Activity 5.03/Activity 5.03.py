# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 01:06:18 2020

@author: PRAJWAL
"""

import pandas as pd

X = pd.read_csv('avila-tr_feats.csv')
y = pd.read_csv('avila-tr_target.csv')

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.regularizers import l2

def build_model(lambda_parameter):
    model_1 = Sequential()
    model_1.add(Dense(10,activation='relu',input_dim=X.shape[1],kernel_regularizer=l2(lambda_parameter)))
    model_1.add(Dense(6,activation='relu',kernel_regularizer=l2(lambda_parameter)))
    model_1.add(Dense(4,activation='relu',kernel_regularizer=l2(lambda_parameter)))
    model_1.add(Dense(1,activation='sigmoid'))
    
    model_1.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return model_1


from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model,shuffle=False)

lambda_parameter = [0.01, 0.5, 1]
epochs = [50, 100]
batch_size = [20]

param_grid = dict(lambda_parameter=lambda_parameter,epochs=epochs,batch_size=batch_size) 

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=model,param_grid=param_grid,cv=5)
results_1 = grid_search.fit(X,y)


print("Best cross-validation score =", results_1.best_score_)
print("Parameters for Best cross-validation score=", results_1.best_params_)
# print the results for all evaluated hyperparameter combinations
accuracy_means = results_1.cv_results_['mean_test_score']
accuracy_stds = results_1.cv_results_['std_test_score']
parameters = results_1.cv_results_['params']
for p in range(len(parameters)):
    print("Accuracy %f (std %f) for params %r" % (accuracy_means[p], accuracy_stds[p], parameters[p]))
 

 
    
    