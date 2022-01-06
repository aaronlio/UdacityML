#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("./ud120projects/tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###


from sklearn import svm

#learn from a subset of data for faster results

#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

clf = svm.SVC(C=10000,kernel='rbf')

t0 = time()
clf.fit(features_train, labels_train)

print("training time:", round(time() - t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print("prediction time:", round(time() - t1, 3), "s")

from sklearn.metrics import accuracy_score

print(accuracy_score(pred, labels_test))

# Should be .984 with a full dataset and .884 with the 1% training set, should be .895 with the rbf kernel (default C), and .899 with C = 10000. 
# Should be > .99 with full dataset

#print(pred[10], pred[26], pred[50]) 

print(list(pred).count(1))

# In general, the SVM is MUCH slower to train and use for predicting than Naive Bayes.

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
