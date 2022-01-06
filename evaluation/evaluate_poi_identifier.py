#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import joblib
import sys

import numpy
sys.path.append("./ud120projects/tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open("./ud120projects/final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = './ud120projects/tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)

### it's all yours from here forward!  
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#Number of POI's in test set
print(labels_test.count(1))
#Number of people in test set
print(len(labels_test))


print("Hypothetical Accuracy: ", 25/29) #Number of true positives "actual non-poi's" vs "actual + wrong non-poi's (4)"

from sklearn.metrics import recall_score, precision_score
#clf = svm.SVC(kernel='linear', C=1).fit(features_train, labels_train)
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print(clf.score(features_test, labels_test))
pred = clf.predict(features_test)

count = 0
for i in range(len(labels_test)):
    if (pred[i] == 1) and (labels_test == 1):
        count += 1 

print(recall_score(labels_test, pred))
print(precision_score(labels_test, pred))
