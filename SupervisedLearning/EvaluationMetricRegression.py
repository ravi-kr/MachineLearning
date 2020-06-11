# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:04:28 2020

@author: Ravi Kumar
"""



import statistics as st
sample = [600, 470, 170, 430, 300] 
print(st.mean(sample))
print(st.pstdev(sample)) 
print(st.pvariance(sample))


from sklearn.metrics import explained_variance_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
explained_variance_score(y_true, y_pred)  

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
explained_variance_score(y_true, y_pred, multioutput='uniform_average')


from sklearn.metrics import max_error
y_true = [3, 2, 7, 1]
y_pred = [4, 2, 7, 1]
max_error(y_true, y_pred)

#___________________evaluation metrics machine learning using linear DataSet________

#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Import the data set from Desktop
dataset = pd.read_csv('Salary_DataSet.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
#Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.3, random_state=0)
#regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
#for predict the test values
y_predict=reg.predict(X_test)
#Visualize the Traing data
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title("linear Regression Salary Vs Experience")
plt.xlabel("Years of Employee")
plt.ylabel("Saleries of Employee")
plt.show()
#Visualize the testing data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title("linear Regression Salary Vs Experience")
plt.xlabel("Years of Employee")
plt.ylabel("Saleries of Employee")
plt.show()

import statsmodels.api as sm
#import statsmodels.formula.api as sm
#import statsmodels.tools.tools.add_constant as sv
X1=sm.add_constant(X)
reg= sm.OLS(y,X1).fit()
reg.summary()

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_predict)


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_predict)

from math import sqrt
from sklearn.metrics import mean_squared_error
result=sqrt(mean_squared_error(y_test, y_predict))

from sklearn.metrics import mean_squared_log_error
np.sqrt(mean_squared_log_error(y_test, y_predict))

import statsmodels.api as sm
#import statsmodels.formula.api as sm
#import statsmodels.tools.tools.add_constant as sv
X1=sm.add_constant(X)
reg= sm.OLS(y,X1).fit()
reg.summary()
