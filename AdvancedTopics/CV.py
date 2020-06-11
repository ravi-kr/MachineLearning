# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:00:39 2020

@author: Ravi Kumar
"""

from sklearn.utils import resample

data = [1, 2, 3, 4, 5, 6]

outputBoot_resample = resample(data, replace=True, n_samples=4, random_state=1)

print('Bootstrap Sample: %s' % outputBoot_resample)

result = [x for x in data if x not in outputBoot_resample]

print('OOB Sample: %s' % result)



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
iris.data.shape, iris.target.shape


X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)

X_train.shape, y_train.shape

X_test.shape, y_test.shape


clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)  


#Leave-One-Out 

import numpy as np
from sklearn.cross_validation import LeaveOneOut

X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

Y = np.array([0, 1, 0, 1])

loo = LeaveOneOut(len(Y))
print(loo)

for train, test in loo:
    print(train, test)
    
    

#Leave-P-Out 

from sklearn.cross_validation import LeavePOut
X = [[0, 0], [1, 1], [2, 2], [3, 3]]

Y = [0, 1, 0, 1]

lpo = LeavePOut(len(Y), 2)
print(lpo)

for train, test in lpo:
    print(train, test)


#KFold

import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
kf.get_n_splits(X)

print(kf)  

for train_index, test_index in kf.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   
   
   
#StratifiedKFold
   
import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

print(skf)  

for train_index, test_index in skf.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   
   
#TimeSeriesSplit
   

from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])

tscv = TimeSeriesSplit(n_splits=3,max_train_size= None)
print(tscv)  

for train, test in tscv.split(X):
    print("%s %s" % (train, test))  
    

'''
 Leave-One-Label-Out - LOLO¶
LeaveOneLabelOut (LOLO) is a cross-validation scheme which holds out the samples according to a third-party provided label.
 This label information can be used to encode arbitrary domain specific stratifications of the samples as integers.

Each training set is thus constituted by all the samples except the ones related to a specific label.

For example, in the cases of multiple experiments, LOLO can be used to create a cross-validation based on the 
different experiments: we create a training set using the samples of all the experiments except one:
'''

from sklearn.cross_validation import LeaveOneLabelOut
X = [[0, 0], [1, 1], [2, 2], [3, 3]]

Y = [0, 1, 0, 1]

labels = [1, 1, 2, 2]

lolo = LeaveOneLabelOut(labels)
print(lolo)

for train, test in lolo:
    print(train, test)
    
    
'''
Leave-P-Label-Out
LeavePLabelOut is similar as Leave-One-Label-Out, but removes samples related to P labels for each 
training/test set.

'''
from sklearn.cross_validation import LeavePLabelOut
X = [[0., 0.], [1., 1.], [-1., -1.], 
     [2., 2.], [3., 3.], [4., 4.]]

Y = [0, 1, 0, 1, 0, 1]

labels = [1, 1, 2, 2, 3, 3]

lplo = LeavePLabelOut(labels, 2)
print(lplo)

for train, test in lplo:
    print(train, test)
    

'''
Random permutations cross-validation a.k.a. Shuffle & Split¶
ShuffleSplit

The ShuffleSplit iterator will generate a user defined number of independent train / test dataset splits. Samples are first shuffled and then splitted into a pair of train and test sets.

It is possible to control the randomness for reproducibility of the results by explicitly seeding the random_state pseudo random number generator.

Here is a usage example:
    
    "Random permutation cross-validator

Yields indices to split data into training and test sets.

Note: contrary to other cross-validation strategies, random splits do not guarantee that all folds will be different, although this is still very likely for sizeable datasets."

'''

import numpy as np
from sklearn.model_selection import ShuffleSplit
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])

y = np.array([1, 2, 1, 2, 1, 2])

rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
rs.get_n_splits(X)

print(rs)

for train_index, test_index in rs.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
 
rs = ShuffleSplit(n_splits=5, train_size=0.5, test_size=.25,
                  random_state=0)

for train_index, test_index in rs.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   
   
   
   
'''

Time Series CV using Python very basic Example to understand the working ...
'''
   
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])

y = np.array([1, 2, 3, 4, 5, 6])

tscv = TimeSeriesSplit(n_splits=5)
print(tscv)  

for train_index, test_index in tscv.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   
   
   
   
   
   
   
  