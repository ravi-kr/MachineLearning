# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:40:08 2020

@author: Ravi Kumar
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Poly_dataSet.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values


'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(np.reshape(y,(10,1)))'''

# Fitting Linear Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
lin_reg = DecisionTreeRegressor(random_state =0)
lin_reg.fit(X, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



# Visualising the Linear Regression results

x_grid=np.arange(min(X),max(X), 0.1)
x_grid =x_grid.reshape(len(x_grid),1)
plt.scatter(X, y, color = 'red')
plt.plot(x_grid, lin_reg.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (DT Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Predict by DT Regression
lin_reg.predict(np.reshape(6.5,(1,1)))


