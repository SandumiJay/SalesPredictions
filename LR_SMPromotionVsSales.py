# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:18:47 2022

@author: Sandumi Jayasekara

Linear Regression Model 

Social Media Promotion Vs Sales
"""

import DataPreProcessing as pross
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression

processed_data=pross.dataPreProcess('Dummy Data HSS.csv','Social Media','Sales')

X_train = processed_data[0]
X_test = processed_data[1]
Y_train = processed_data[2]
Y_test = processed_data[3]

# print(X_train[:,4:5])


np.isnan(Y_train).any()
     
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Social Media Promotion vs Sales (Training set)')
plt.xlabel('Social Media Promotion (millions)')
plt.ylabel('Sales')
plt.show()


plt.scatter(X_test, Y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Social Media Promotion vs Sales (Test set)')
plt.xlabel('Social Media Promotion (millions)')
plt.ylabel('Sales')
plt.show()

print("********************Regression Equation **********************")


a=str(regressor.coef_ [0][0])
b = str(regressor.intercept_[0])


print("Y = "+  a+"X + " +b)


# Save build model

with open('SalesPredfromSMpromotion_model_pkl', 'wb') as files:
    pickle.dump(regressor, files)
