# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:29:36 2022

@author: Sandumi Jayasekara
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_raw = pd.read_csv('Dummy Data HSS.csv')

print(data_raw.head())

#convert categorical data into indicator variable 
data = pd.get_dummies(data_raw, drop_first = True)

print("*****************data with coverted categorical data******************")
print(data.head())

# re-order the columns  
data = data[['TV', 'Radio', 'Social Media', 'Influencer_Mega', 'Influencer_Micro', 'Influencer_Nano', 'Sales']]


print("****************Shape of the dataset**********************")
print(data.shape)

#Find missing values

missing_values = data.isnull().sum()
print(missing_values)

#Fill missing Values

data = data.fillna(data.mean())

#Summary of the data

print("****************Summary of the dataset**********************")

data_summary =data.describe().T
print(data_summary)

data_summary.to_csv('data_summary.csv')

#Data Visulaization
data.hist()
plt.savefig('plots/Data Distribution-Histrogram.png')

sns.pairplot(data = data,x_vars = ['TV', 'Radio', 'Social Media'],y_vars = 'Sales',size = 7,kind = 'reg')


plt.figure(figsize = (10,7))
sns.scatterplot(data = data, 
                y = 'Sales', 
                x = 'Radio', 
                hue = data_raw['Influencer'])
plt.title('Sales Distribution')
plt.savefig('plots/Sales Distribution-Radio.png')
plt.show()


plt.figure(figsize = (10,7))
sns.scatterplot(data = data, 
                y = 'Sales', 
                x = 'TV', 
                hue = data_raw['Influencer'])
plt.title('Sales Distribution')
plt.savefig('plots/Sales Distribution-TV.png')
plt.show()


plt.figure(figsize = (10,7))
sns.scatterplot(data = data, 
                y = 'Sales', 
                x = 'Social Media', 
                hue = data_raw['Influencer'],
               size = data_raw['Influencer'])
plt.title('Sales Distribution')
plt.savefig('plots/Sales Distribution-Social Media.png')
plt.show()
