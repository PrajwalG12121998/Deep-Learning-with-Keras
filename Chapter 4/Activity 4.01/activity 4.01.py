# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:42:36 2020

@author: PRAJWAL
"""
import pandas as pd
X = pd.read_csv('HCV_feats.csv')
y = pd.read_csv('HCV_target.csv')

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from keras.models import Sequential
from keras.layers import Dense,Activation

def build_model():
    model = Sequential()
    model.add(Dense(4,activation='tanh',input_dim=28))
    model.add(Dense(2,activation='tanh'))
    model.add(Dense(1,activation='sigmoid'))
    #compile model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    #return model
    return model

from keras.wrappers.scikit_learn import KerasClassifier

YourModel = KerasClassifier(build_fn=build_model,epochs=100,batch_size=20,verbose=1,shuffle=False)
    
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5)

from sklearn.model_selection import cross_val_score
results = cross_val_score(YourModel,X,y,cv=kf)

#print accuracy for each fold
for f in range(0,5):
    print("Test accuracy at fold", f+1," = ",results[f])
    
print("\n")

# print the result
print(f"Final Cross-Validation Test accuracy = {abs(results.mean()):.4f}")
#Ans = 0.51



    