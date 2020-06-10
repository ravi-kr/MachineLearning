# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:41:55 2020

@author: Ravi Kumar
"""

#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the data set from Desktop
dataset = pd.read_csv('M_Regression.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.3, random_state=0)

#regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

#for predict the test values
y_predict=reg.predict(X_test)




