# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:29:21 2020

@author: Ravi Kumar
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('KMeans.csv')
X = dataset.iloc[:, [1, 2]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Customer')
plt.ylabel('Distances')
plt.show();

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity ='euclidean', linkage = 'ward' )
y_hc=hc.fit_predict(X)
