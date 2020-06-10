# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:11:39 2020

@author: Ravi Kumar
"""


#import the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



#Import the data set

dataset = pd.read_csv('DataSet.csv')

X=dataset.iloc[:,:-1].values
XX=dataset.iloc[:,:-1].values
XXX=dataset.iloc[:,:-1].values

y=dataset.iloc[:,3].values



#Missing Value Handling by MEAN

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer=imputer.fit(X[:,1:3])

X[:,1:3]= imputer.transform(X[:,1:3])



#Missing Value Handling by MEDIAN

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='median')


imputer=imputer.fit(XX[:,1:3])

XX[:,1:3]= imputer.transform(XX[:,1:3])



#Missing Value Handling by most_frequent

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer=imputer.fit(XXX[:,1:3])

XXX[:,1:3]= imputer.transform(XXX[:,1:3])


#Concept of Dummy Variable, Handling the conflict of them

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()

X[:,0]=labelencoder_X.fit_transform(X[:,0])

onehotencoder = ColumnTransformer([("Developer", OneHotEncoder(), [0])], remainder = 'passthrough')

X = onehotencoder.fit_transform(X)



#Training and Testing Data (divide the data into two part)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, random_state=0)



#Standard and fit the data for better predication 

from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

X_test=sc_X.fit_transform(X_test)


X_train=sc_X.fit_transform(X_train)

