# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:13:54 2020

@author: Ravi Kumar
"""

#import the libraries
import numpy as py
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
plt.ylabel("Salary of Employee")
plt.show()

#Visualize the testing data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title("linear Regression Salary Vs Experience")
plt.xlabel("Years of Employee")
plt.ylabel("Salary of Employee")
plt.show()



