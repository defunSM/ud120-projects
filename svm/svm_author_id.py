#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

clf = svm.SVC(kernel='linear')
t0 = time()
y_pred = clf.fit(features_train, labels_train).predict(features_test)

print("training time:", round(time()-t0, 3), "s")
print("Number of mislabeled points out of a total %d points : %d" % (features_train.shape[0], (labels_train != y_pred).sum()))






#########################################################
