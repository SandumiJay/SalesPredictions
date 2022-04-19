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


def dataPreProcess(filepath,X_column,Y_column):
    
    data = pd.read_csv(filepath)

    for col in data.columns:
        if(col==X_column or col==Y_column):
            continue
        else:
            data=data.drop(col, 1)
     
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    
    
    print(Y.iloc[:])
    Y=Y.iloc[:].values.reshape(-1, 1)
    imputer = SimpleImputer(missing_values= np.nan, strategy ='mean')
    imputer.fit(Y)
    Y=imputer.transform(Y)
    print("Missing Y Values replaced !!!!!!!")
    
    
    X=X.iloc[:].values.reshape(-1, 1)
    if(X_column != 'Influencer'):
        imputer = SimpleImputer(missing_values= np.nan, strategy ='mean')
        imputer.fit(X)
        X=imputer.transform(X)
        print("Missing X Values replaced !!!!!!!")
        
    else :
        imputer = SimpleImputer(missing_values= np.nan, strategy ='most_frequent')
        imputer.fit(X)
        X=imputer.transform(X)
        print("Missing X Values replaced !!!!!!!")
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))
        print("Categorycal Data Encoded !!!!")
    

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
    
    print("Splited !!!!")
    
    return X_train, X_test, Y_train, Y_test;