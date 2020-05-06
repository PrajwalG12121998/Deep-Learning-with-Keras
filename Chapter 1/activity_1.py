# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:48:42 2020

@author: PRAJWAL
"""
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('online_shoppers_intention.csv')


dataset.dtypes
bounds = dataset.describe()

%matplotlib inline
dataset['Weekend'].value_counts().plot(kind="bar")

dataset['is_weekend'] = dataset['Weekend'].apply(lambda row:1 if row==True else 0)
dataset.drop('Weekend', axis=1,inplace = True)

dataset['Revenue'] = dataset['Revenue'].apply(lambda row:1 if row==True else 0)

colname = 'VisitorType'
visitor_type_dummies = pd.get_dummies(dataset[colname],prefix = colname)
pd.concat([dataset[colname],visitor_type_dummies],axis = 1).tail(10)
visitor_type_dummies.drop('VisitorType_Other',axis=1, inplace=True)
visitor_type_dummies.head()

dataset = pd.concat([dataset, visitor_type_dummies],axis=1)
dataset.drop('VisitorType',axis=1, inplace=True)

colname = 'Month'
dataset['Month'].value_counts().plot(kind='bar')

month_dummies = pd.get_dummies(dataset[colname],prefix=colname)
dataset = pd.concat([dataset,month_dummies],axis=1)

dataset.drop('Month',axis=1,inplace=True)

feature = dataset.drop('Revenue', axis=1)
target = dataset['Revenue']



dataset.to_csv('C:\Deep_Learning\Activities\Activity 1\OSI_dataset_e2.csv',index=False)
feature.to_csv('C:\Deep_Learning\Activities\Activity 1\OSI_feature_e2.csv',index=False)
target.to_csv('C:\Deep_Learning\Activities\Activity 1\OSI_target_e2.csv',index=False)

#Traffic Type is a categorized variable
colname='TrafficType'
traffic_type_dummies = pd.get_dummies(dataset[colname],prefix=colname)

dataset[colname].value_counts()
traffic_type_dummies.drop(colname+'_17',axis=1,inplace=True)
dataset = pd.concat([dataset,traffic_type_dummies],axis=1)
dataset.drop('TrafficType',axis=1,inplace=True)

#Operating System is a categorized variable
colname='OperatingSystems'
operatingSystems_dummies = pd.get_dummies(dataset[colname],prefix=colname)

dataset[colname].value_counts()
operatingSystems_dummies.drop(colname+'_5',axis=1,inplace=True)
dataset = pd.concat([dataset,operatingSystems_dummies],axis=1)
dataset.drop(colname,axis=1,inplace=True)

#Browser is a categorized variable
colname='Browser'
browser_dummies = pd.get_dummies(dataset[colname],prefix=colname)

dataset[colname].value_counts()
browser_dummies.drop(colname+'_9',axis=1,inplace=True)
dataset = pd.concat([dataset,browser_dummies],axis=1)
dataset.drop(colname,axis=1,inplace=True)

#Region is a categorized variable
colname='Region'
region_dummies = pd.get_dummies(dataset[colname],prefix=colname)

dataset[colname].value_counts()
region_dummies.drop(colname+'_5',axis=1,inplace=True)
dataset = pd.concat([dataset,region_dummies],axis=1)
dataset.drop(colname,axis=1,inplace=True)

feature = dataset.drop('Revenue', axis=1)
target = dataset['Revenue']

dataset.to_csv('C:\Deep_Learning\Activities\Activity 1\OSI_dataset_e3.csv',index=False)
feature.to_csv('C:\Deep_Learning\Activities\Activity 1\OSI_feature_e3.csv',index=False)
target.to_csv('C:\Deep_Learning\Activities\Activity 1\OSI_target_e3.csv',index=False)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(feature,target,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_pred=y_pred,y_true=y_test)

precision, recall, fscore, _ = metrics.precision_recall_fscore_support(y_pred=y_pred,y_true=y_test, average='binary')

coef_list = [f'{feature}: {coef}' for coef, feature in sorted(zip(model.coef_[0], X_train.columns.values.tolist()))]
for item in coef_list:
    print(item)

conf_matrix = metrics.confusion_matrix(y_test, y_pred)

#baseline model
target = pd.read_csv('OSI_target_e3.csv')
target['0'].value_counts()/12330

y_baseline = pd.Series(data=[0]*target.shape[0])

precision, recall, fscore, _ = metrics.precision_recall_fscore_support(y_pred=y_baseline,y_true=target['0'], average='macro')


