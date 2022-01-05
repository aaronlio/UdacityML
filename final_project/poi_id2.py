import sys
import pickle
import numpy as np
import pandas as pd
import sklearn
#from ggplot import *
import matplotlib as plt

sys.path.append("./tools/")
import tester

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
### Load the dictionary containing the dataset
with open("final_project/final_project_dataset_modified.pkl", "rb") as data_file:
    enron_dict = pickle.load(data_file, encoding='ASCII')
    
enron_dict['METTS MARK'].keys()

f = open('final_project/poi_names.txt', 'r')

# Change data dictionary to pandas DataFrame
df = pd.DataFrame.from_records(list(enron_dict.values()))
persons = pd.Series(list(enron_dict.keys()))

print(persons.head())
print(df.head())

# convert to numpy.nan
df.replace(to_replace='NaN', value=np.nan, inplace=True)

# count number of nan for columns
print(df.isnull().sum())

# remove column from df if null counter > 65
for column, series in df.iteritems():
    if series.isnull().sum() > 65:
        df.drop(column, axis=1, inplace=True)

# remove email address column
if 'email_address' in list(df.columns.values):
    df.drop('email_address', axis=1, inplace=True)
    
    
# Impute the missing values
#from sklearn.preprocessing import Imputer
#imp = Imputer(missing_values='NaN', strategy='median', axis=0)
#imp.fit(df)
#df_imp = pd.DataFrame(imp.transform(df.copy(deep=True)))

df_imp = df.replace(to_replace=np.nan, value=0)
df_imp = df.fillna(0).copy(deep=True)
df_imp.columns = list(df.columns.values)

print(df_imp.isnull().sum())
print(df_imp.head())

df_imp.describe()

# drop row for 'THE TRAVEL AGENCY IN THE PARK'

park_index = list(enron_dict.keys()).index('THE TRAVEL AGENCY IN THE PARK')
print(park_index)
df_imp_sub = df_imp.drop(df_imp.index[[park_index]])
print(df_imp_sub['total_payments'].max())
enron_dict['LOCKHART EUGENE E']

lockhart_index = list(enron_dict.keys()).index('LOCKHART EUGENE E')
print(lockhart_index)
df_imp_sub = df_imp.drop(df_imp.index[[lockhart_index]])

# Rename datafram name after cleaning up NaN and outlier
enron_df = df_imp_sub

# Graph for features

import matplotlib.pyplot as plt

plt.scatter(enron_df['salary'], enron_df['total_payments'])
plt.xlabel("salary")
plt.ylabel("total_payments")
plt.show()

plt.scatter(enron_df['salary'], enron_df['total_stock_value'])
plt.xlabel("salary")
plt.ylabel("total_stock_value")
plt.show()

enron_df['salary_of_total_payments'] = 0.0
enron_df['salary_of_total_stock_value'] = 0.0
enron_df.loc[enron_df['total_payments'] != 0.0,'salary_of_total_payments'] = enron_df['salary'] / enron_df['total_payments'] * 100
enron_df.loc[enron_df['total_stock_value'] != 0.0,'salary_of_total_stock_value'] = enron_df['salary'] / enron_df['total_stock_value'] * 100

# Graph 'salary_of_total_payment' and 'salary_of_total_stock_value' to salary
plt.scatter(enron_df['salary'], enron_df['salary_of_total_payments'], color='blue')
plt.scatter(enron_df['salary'], enron_df['salary_of_total_stock_value'], color='red')
plt.xlabel('Salary')
plt.ylabel('Of Total Payment / Of Total Stock Value')
plt.show()
enron_df['poi_ratio'] = 0.0
enron_df['fraction_to_poi'] = 0.0
enron_df['fraction_from_poi'] = 0.0

enron_df.loc[(enron_df['from_messages'] + enron_df['to_messages']) != 0.0, 'poi_ratio'] = (enron_df['from_poi_to_this_person'] + enron_df['from_this_person_to_poi']) / (enron_df['from_messages'] + enron_df['to_messages']) * 100
enron_df.loc[enron_df['from_messages'] != 0.0, 'fraction_to_poi'] = enron_df['from_this_person_to_poi'] / enron_df['from_messages'] * 100
enron_df.loc[enron_df['to_messages'] != 0.0, 'fraction_from_poi']= enron_df['from_poi_to_this_person'] / enron_df['to_messages'] * 100
# Graph 'fraction_to_poi' and 'fraction_from_poi' to salary
plt.scatter(enron_df['salary'], enron_df['fraction_to_poi'], color='blue')
plt.scatter(enron_df['salary'], enron_df['fraction_from_poi'], color='red')
plt.xlabel('Salary')
plt.ylabel('Fraction')
plt.show()

# move 'poi' to the first column
cols = enron_df.columns.tolist()
#print cols
cols = cols[7:8] + cols[:7] + cols[8:]
#print cols
enron_df = enron_df[cols]
#print enron_df.columns.values

predictors = enron_df.drop('poi', axis=1).columns.values

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(enron_df[predictors], enron_df["poi"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

indices = np.argsort(scores)[::-1]

sorted_important_features=[]
for i in indices:
    sorted_important_features.append(predictors[i])

# Plot the scores.  
plt.figure()
plt.title("Feature Importances")
plt.bar(range(np.size(predictors)), scores[indices],
       color="seagreen", yerr=np.std([indices]), align="center")
plt.xticks(range(np.size(predictors)), sorted_important_features, rotation='vertical')

plt.xlim([-1, np.size(predictors)])
plt.show()