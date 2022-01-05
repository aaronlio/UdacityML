#!/usr/bin/python

import sys
from numpy.lib.arraypad import pad
from sklearn.decomposition import PCA
import pickle
import numpy as np
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import *
import sklearn.metrics
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from poi_id_helper_functions import outlierCleaner
from tester import dump_classifier_and_data
np.warnings.filterwarnings('ignore')

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
'long_term_incentive', 'restricted_stock', 'director_fees',
'to_messages', 'from_poi_to_this_person', 
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project/final_project_dataset_modified.pkl", "rb") as data_file:
    enron_dict = pickle.load(data_file, encoding='ASCII')
    
enron_dict['METTS MARK'].keys()

f = open('final_project/poi_names.txt', 'r')
# Number of people in the dataset = 146
#Number of initial features = 21
panda_data = pd.DataFrame.from_records(list(enron_dict.values()))
persons = pd.Series(list(enron_dict.keys()))



panda_data.replace(to_replace='NaN', value=np.nan, inplace=True)

pd.set_option("display.max_rows", None, "display.max_columns", None)


## Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# Based on the lack of information provided by the users email (for this specific project), it can be removed as a feature
# bringing the total down to 20. More importantly, the rest of the features have values which are float64, so using both 
# text and numeric data may overcomplicate the project
# remove email address column

# Futhermore, if more than 50% of the dataset is null for a particular category, it will be removed, simply because there
# is not enough data available to make it a useful feature.



for column, row in panda_data.iteritems():
    if row.count() < 65:
        panda_data.drop(column, axis=1, inplace=True)

if 'email_address' in list(panda_data.columns.values):
    panda_data.drop('email_address', axis=1, inplace=True)


panda_data.replace(to_replace=np.nan, value = 0, inplace=True)
panda_data = panda_data.fillna(0).copy(deep=True)

#print(panda_data.describe())
bad_entry1 = list(enron_dict.keys()).index('THE TRAVEL AGENCY IN THE PARK')
panda_data = panda_data.drop(panda_data.index[[bad_entry1]])
bad_entry2 = list(enron_dict.keys()).index('LOCKHART EUGENE E')
panda_data = panda_data.drop(panda_data.index[[bad_entry2]])
print(panda_data['total_payments'].max())


my_dataset = panda_data
my_dataset = my_dataset[ ['poi'] + [ col for col in my_dataset.columns if col != 'poi' ]]

import matplotlib.pyplot as plt
plt.subplot(2, 2, 1)
plt.scatter(my_dataset['salary'], my_dataset['bonus'])
plt.xlabel("salary")
plt.ylabel("bonus")


plt.subplot(2,2,2)
plt.scatter(my_dataset['salary'], my_dataset['total_payments'])
plt.xlabel("salary")
plt.ylabel("total_payments")

# Creating 2 new financial features, salary as a ratio of total_payments and bonuses
my_dataset.loc[:, 'salary_percentage_of_total'] = 0.0
my_dataset.loc[:, 'salary_percentage_of_bonuses'] = 0.0
my_dataset.loc[my_dataset['total_payments'] != 0.0, 'salary_percentage_of_total'] = my_dataset['salary']/my_dataset['total_payments'] * 100
my_dataset.loc[my_dataset['bonus'] != 0.0, 'salary_percentage_of_bonuses'] = my_dataset['salary']/my_dataset['bonus'] * 100


plt.subplot(2,2,3)
plt.scatter(my_dataset['salary'], my_dataset['salary_percentage_of_total'], color='green')
plt.scatter(my_dataset['salary'], my_dataset['salary_percentage_of_bonuses'], color='red')
plt.xlabel('Salary')
plt.ylabel('Salary as a % Of Total Payment / Bonuses')


# Creating additional email-related features introduced by Kate in the course. She hypothesized that correspondences
# between POI's should be higher than between POI's and non-POI's. A feature will be created that combines messages to and from
# POI's
my_dataset.loc[:, 'fraction_to_poi'] = 0.0
my_dataset.loc[:,'fraction_from_poi'] = 0.0
my_dataset.loc[:, 'Percentage_of_interactions_with_POI'] = 0.0
my_dataset.loc[((my_dataset['to_messages'] + my_dataset['from_messages']) != 0.0), 'Percentage_of_interactions_with_POI'] = \
    ((my_dataset['from_poi_to_this_person'] + my_dataset['from_this_person_to_poi'])/(my_dataset['to_messages'] + my_dataset['from_messages']))

my_dataset.loc[my_dataset['from_messages'] != 0.0, 'fraction_to_poi'] = my_dataset['from_this_person_to_poi'] / my_dataset['from_messages'] * 100
my_dataset.loc[my_dataset['to_messages'] != 0.0, 'fraction_from_poi']= my_dataset['from_poi_to_this_person'] / my_dataset['to_messages'] * 100

# Potential for Salary Scaling between zero and one, though not necessary
#max_salary = my_dataset['salary'].max()
#min_salary = my_dataset['salary'].min()
#my_dataset.loc[(my_dataset['salary'] != 0.0), 'salary'] = (my_dataset['salary']-min_salary)/(max_salary-min_salary)

plt.subplot(2,2,4)
plt.scatter(my_dataset['salary'], my_dataset['Percentage_of_interactions_with_POI'], color='purple')

plt.xlabel('Scaled salary')
plt.ylabel(f'% of interactions with POIs')
plt.autoscale(enable=True) 
plt.tight_layout()
#plt.show()

features_to_use = my_dataset.drop('poi', axis=1).columns.values

### Extract features and labels from dataset for local testing
# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(my_dataset[features_to_use], my_dataset["poi"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

indices = np.argsort(scores)[::-1]

sorted_important_features=[]
for i in indices:
    sorted_important_features.append(features_to_use[i])

# Plot the scores.  
plt.figure()
plt.title("Feature Importances")
plt.bar(range(np.size(features_to_use)), scores[indices],
       color="seagreen", yerr=np.std([indices]), align="center")
plt.xticks(range(np.size(features_to_use)), sorted_important_features, rotation=45, ha = 'right')
plt.tight_layout()
plt.xlim([-1, np.size(features_to_use)])
#plt.show()

# Based on the results presented above, we can classify the most important features, for example, the five most important are:
three_important_predictors = ['bonus', 'exercised_stock_options', 'total_stock_value']
five_important_predictors = ['bonus', 'exercised_stock_options', 'total_stock_value', 'fraction_to_poi', 'salary']
ten_important_predictors = ['bonus', 'exercised_stock_options', 'total_stock_value', 'fraction_to_poi', 'salary', 'shared_receipt_with_poi', 'Percentage_of_interactions_with_POI', 'expenses', 'from_poi_to_this_person', 'salary_percentage_of_total']
predictors = ['salary_percentage_of_bonuses','other','from_messages','total_payments','to_messages','restricted_stock','from_this_person_to_poi','fraction_from_poi','bonus', 'exercised_stock_options', 'total_stock_value', 'fraction_to_poi', 'salary', 'shared_receipt_with_poi', 'Percentage_of_interactions_with_POI', 'expenses', 'from_poi_to_this_person', 'salary_percentage_of_total']

# We must also split the data into labels and features
labels = (my_dataset['poi']).copy(deep=True).astype(int)
labels = labels.to_numpy()
print(labels)

features = (my_dataset[predictors]).fillna(0).copy(deep=True).to_numpy()



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
Initial_param_grid_SVC = {

         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          'degree': [2,3,4]
          }
listt = []
for i in range(2, 5):
    listt.append(i)

listt2 = []
for i in range(4, 16):
    listt2.append(i)

listt3 = []
for i in range(4, 125):
    listt3.append(i)

initial_param_grid_RFS = {
    'criterion': ['gini', 'entropy'],
    'min_samples_split': listt,
    'min_samples_leaf' : [1,2,3],
    'random_state': [0, 5, 42],
    'n_estimators': listt3,
    'max_depth': [None]

          }

initial_param_grid_ETS = {
    'min_samples_split': listt,
    'min_samples_leaf' : [1,2,3],
    'random_state': [5, 10, 25, 45, 60, 75, 90, 95, 100, 105, 120],
          }

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

#<------------------------- Initial testing different classifiers -------------------------->

clf1_NB = GaussianNB()
score_3 = cross_val_score(clf1_NB, my_dataset[three_important_predictors], labels)
score_5 = cross_val_score(clf1_NB, my_dataset[five_important_predictors], labels)
score_10 = cross_val_score(clf1_NB, my_dataset[ten_important_predictors], labels)
print(f"Naive Bayes:\n3 Predictors Score: {mean(score_3)}\n5 Predictors Score: {mean(score_5)}\n10 Predictors Score: {mean(score_10)}\n")

svc = SVC()
clf2_SVM = GridSearchCV(svc, Initial_param_grid_SVC)
score_3 = cross_val_score(clf2_SVM, my_dataset[three_important_predictors], labels)
score_5 = cross_val_score(clf2_SVM, my_dataset[five_important_predictors], labels)
score_10 = cross_val_score(clf2_SVM, my_dataset[ten_important_predictors], labels)
print(f"SVM:\n3 Predictors Score: {mean(score_3)}\n5 Predictors Score: {mean(score_5)}\n10 Predictors Score: {mean(score_10)}\n")


dt = ExtraTreeClassifier()
clf3_ETC = GridSearchCV(dt, initial_param_grid_ETS)
score_3 = cross_val_score(clf3_ETC, my_dataset[three_important_predictors], labels)
score_5 = cross_val_score(clf3_ETC, my_dataset[five_important_predictors], labels)
score_10 = cross_val_score(clf3_ETC, my_dataset[ten_important_predictors], labels)
print(f"Extra Tree:\n3 Predictors Score: {mean(score_3)}\n5 Predictors Score: {mean(score_5)}\n10 Predictors Score: {mean(score_10)}\n")


clf4_RFC = RandomForestClassifier()
#clf4_RFC = GridSearchCV(rf, param_grid_RFS)
score_3 = cross_val_score(clf4_RFC, my_dataset[three_important_predictors], labels)
score_5 = cross_val_score(clf4_RFC, my_dataset[five_important_predictors], labels)
score_10 = cross_val_score(clf4_RFC, my_dataset[ten_important_predictors], labels)
print(f"Random Forest:\n3 Predictors Score: {mean(score_3)}\n5 Predictors Score: {mean(score_5)}\n10 Predictors Score: {mean(score_10)}\n")

clf5_ABC = AdaBoostClassifier()
score_3 = cross_val_score(clf5_ABC, my_dataset[three_important_predictors], labels)
score_5 = cross_val_score(clf5_ABC, my_dataset[five_important_predictors], labels)
score_10 = cross_val_score(clf5_ABC, my_dataset[ten_important_predictors], labels)
print(f"Adaboost:\n3 Predictors Score: {mean(score_3)}\n5 Predictors Score: {mean(score_5)}\n10 Predictors Score: {mean(score_10)}\n")

#<------------------------- SVM, Extra Tree, and Random Forest -------------------------->

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
n_components = 8

# Example starting point. Try investigating other evaluation techniques!

""" pca = PCA(n_components=n_components, whiten=True)
selection = SelectKBest(k=4)

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
X_features = combined_features.fit(features, labels).transform(features)

svm = SVC(kernel='linear')
# Do grid search over k, n_components and C:
pipeline = Pipeline([("features", combined_features), ("svm", svm)])


param_grid_svm = dict(n_components=[1, 2, 3, 4, 5],
                  select_k=[1, 2,3],
                  C=[.001, .01, 0.1, 1, 10],
                  gamma = [1e3, 5e3, 1e4, 5e4, 1e5, 'auto'])


grid_search = GridSearchCV(pipeline, param_grid=param_grid_svm, verbose=10)
grid_search.fit(features, labels)
print(grid_search.best_estimator_)
"""
#<------------------------- SVC Improved -------------------------->
# Create a pipeline that extracts features from the data using PCA and SelectKBest then creates a model_SVC
# create feature union
features_pipeline = []
features_pipeline.append(('pca', PCA(n_components=4)))
features_pipeline.append(('select_best', SelectKBest(k=8)))
feature_union = FeatureUnion(features_pipeline)

estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('svc', SVC(kernel='rbf', class_weight='balanced')))
model_svc = Pipeline(estimators)

param_grid_svm = {
            'svc__C': [1e3, 5e3, 1e4, 5e4, 1e5],
            'svc__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1 ],
            'svc__degree': [1, 2, 3, 4, 5]
            }

grid_svc = GridSearchCV(model_svc, param_grid_svm)
grid_svc = grid_svc.fit(features, labels)
print(grid_svc.best_estimator_)

# best fitting svc_clf
best_svc_pipe = grid_svc.best_estimator_ 

# evaluate pipeline for rbfSVC
seed = 100
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
scores_svc = cross_val_score(model_svc, my_dataset[predictors], labels, cv=kfold)
print('rbfSVC mean score:', mean(scores_svc))



#<------------------------- Random Forest Improvements -------------------------->
# create pipeline for RandomForest
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('RandomForest', RandomForestClassifier()))
model_rf = Pipeline(estimators)

param_grid = {'RandomForest__n_estimators': [1,2,3,4,5,10,20,30,40,50,100],
               'RandomForest__min_samples_split' : listt,
               'RandomForest__min_samples_leaf' : [1,2,3,4]
             }

grid_rf = GridSearchCV(model_rf, param_grid)
grid_rf = grid_rf.fit(features, labels)
print(grid_rf.best_estimator_)

scores_rf = cross_val_score(model_rf, my_dataset[predictors], labels)
print('RandomForest mean score: ', mean(scores_rf))

#<------------------------- Extra Tree Improvements -------------------------->
# create pipeline for Extra Tree
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('ExtraTree', ExtraTreeClassifier()))
model_et = Pipeline(estimators)

param_grid = {
    'ExtraTree__min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    'ExtraTree__min_samples_leaf' : [1,2,3],
    'ExtraTree__random_state': [1, 5, 10, 25, 42, 45, 50, 55, 60, 75, 90, 95, 100, 105, 120],
          }

grid_et = GridSearchCV(model_et, param_grid)
grid_et = grid_et.fit(features, labels)
print(grid_et.best_estimator_)

scores_et = cross_val_score(model_et, my_dataset[predictors], labels)
print('Extra Tree mean score: ', mean(scores_et))


#<---------------------------- FINAL SVC IMPROVEMENTS ---------------------------->
n_comp = 4
pca = PCA(n_components=n_comp).fit(features)
features_transformed = pca.transform(features)
features_pca = pca.components_
param_grid = {
            'C': [1e3, 5e3, 1e4, 5e4, 1e5],
            'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1 ],
            'degree': [1, 2, 3, 4, 5]
}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(features_transformed, labels)
pred_svc = clf.predict(features_transformed)

best_svc_clf = clf.best_estimator_ 
scores = cross_val_score(best_svc_clf, features, labels)
print('SVC after PCA mean score:', mean(scores))

#<---------------------------- FINAL RF IMPROVEMENTS ---------------------------->

parameters = {
    'n_estimators': [10,20,30,40,50,100],
    'min_samples_split': [2,3,4,5,6],
    'min_samples_leaf' : [1,2,3],
    'random_state': [5, 10, 25, 45, 60, 75, 90, 95, 100, 105, 120],
          }

rf_clf = RandomForestClassifier()
grid_obj = GridSearchCV(rf_clf, parameters)
grid_fit = grid_obj.fit(features, labels)

best_rf_clf = grid_fit.best_estimator_ 

best_rf_clf.fit(features,labels)

scores = cross_val_score(best_rf_clf, features, labels)
print('RandomForest mean score:', mean(scores))

#<---------------------------- FINAL ET IMPROVEMENTS ---------------------------->

parameters = { 'min_samples_split' :[2,3,4,5,6],
               'min_samples_leaf' : [1,2,3],
               'random_state': [5, 10, 25, 45, 60, 75, 90, 95, 100, 105, 120]
             }

et_clf = ExtraTreeClassifier()
grid_obj = GridSearchCV(et_clf, parameters)
grid_fit = grid_obj.fit(features, labels)

best_et_clf = grid_fit.best_estimator_ 

best_et_clf.fit(features,labels)

scores = cross_val_score(best_et_clf, features, labels)
print('ExtraTree mean score:', mean(scores))

#<--------------------------------------------- Validation --------------------------------------------->
features_train, features_test, labels_train, labels_test = train_test_split(features_transformed, labels, test_size=0.3, random_state=42)
pred = best_rf_clf.predict(features_test)
acc = sklearn.metrics.accuracy_score(labels_test, pred)

print("accuracy after tuning = ", acc)

# function for calculation ratio of true positives
# out of all positives (true + false)
print('precision = ', sklearn.metrics.precision_score(labels_test,pred))

# function for calculation ratio of true positives
# out of true positives and false negatives
print('recall = ', sklearn.metrics.recall_score(labels_test,pred))
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### generates the necessary .pkl files for validating your results.
### that the version of poi_id.py that you submit can be run on its own and
clf_dump = best_rf_clf


pickle.dump(clf_dump, open("my_classifier.pkl", "w") )
pickle.dump(my_dataset, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )