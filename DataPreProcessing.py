# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 22:20:02 2022

@author: Sandumi Jaysekara
"""

import numpy as np
import matplotlib as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def dataPreProcess(filepath):
    
    data = pd.read_csv(filepath)
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    imputer = SimpleImputer(missing_values= np.nan, strategy ='mean')
    imputer.fit(X.iloc[:,0:3])
    X.iloc[:,0:3]=imputer.transform(X.iloc[:,0:3])
    
    
    imputer2 = SimpleImputer(missing_values= np.nan, strategy ='most_frequent')
    imputer2.fit(X.iloc[:,3:])
    X.iloc[:,3:]=imputer2.transform(X.iloc[:,3:])
    
    print("Missing Values replaced !!!!!!!")
    
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    
    print("Encoded !!!!")
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
    
    print("Splited !!!!")
    
    sc = StandardScaler()
    X_train[:,4:7] = sc.fit_transform(X_train[:,4:7])
    X_test[:, 4:7] = sc.transform(X_test[:, 4:7]) 
    
    print("Transformed to same scale !!!!")
    
    return X_train, X_test, Y_train, Y_test;