# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 22:20:02 2022

@author: User
"""

import numpy as np
import matplotlib as plt
import pandas as pd

from sklearn.impute import SimpleImputer

data = pd.read_csv('Dummy Data HSS.csv')
print(data)

X = data.iloc[:,:-1]
print(X)

Y = data.iloc[:,-1]
print(Y)


imputer = SimpleImputer(missing_values= np.nan, strategy ='mean')
imputer.fit(X.iloc[:,0:3])
X.iloc[:,0:3]=imputer.transform(X.iloc[:,0:3])


imputer2 = SimpleImputer(missing_values= np.nan, strategy ='most_frequent')
imputer2.fit(X.iloc[:,3:])
X.iloc[:,3:]=imputer2.transform(X.iloc[:,3:])

print("Missing Values replaced !!!!!!!")
print(X.iloc[:,3])