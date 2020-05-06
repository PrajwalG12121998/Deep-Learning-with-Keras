# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:34:54 2020

@author: PRAJWAL
"""

#Activity 1
import pandas as pd


dataset = pd.read_csv('OSI_dataset_e3.csv')
feature = pd.read_csv('OSI_feature_e3.csv')
target = dataset['Revenue'] 


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(feature,target,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegressionCV
Cs = np.logspace(-2,6,9)

model_l1 = LogisticRegressionCV(Cs =Cs, penalty = 'l1', cv=10, solver='liblinear', random_state=42)
model_l2 = LogisticRegressionCV(Cs =Cs, penalty = 'l2', cv=10,max_iter=1000, random_state=42)


model_l1.fit(X_train,y_train)
model_l2.fit(X_train,y_train)

l1_pred = model_l1.predict(X_test)
l2_pred = model_l2.predict(X_test)

from sklearn import metrics
accuracy_l1 = metrics.accuracy_score(y_pred=l1_pred,y_true=y_test)
accuracy_l2 = metrics.accuracy_score(y_pred=l2_pred,y_true=y_test)

precision_l1, recall_l1, fscore_l1, _ = metrics.precision_recall_fscore_support(y_pred=l1_pred,y_true=y_test, average='binary')
precision_l2, recall_l2, fscore_l2, _ = metrics.precision_recall_fscore_support(y_pred=l2_pred,y_true=y_test, average='binary')


