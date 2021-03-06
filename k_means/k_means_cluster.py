#!/usr/bin/python3

""" 
    Skeleton code for k-means clustering mini-project.
"""


from math import inf
import joblib
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("./ud120projects/tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = joblib.load( open("./ud120projects/final_project/final_project_dataset.pkl", "rb") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

max_salary = float(-inf)
min_salary = float(inf)
for k in data_dict.keys():
    if float(data_dict[k]["salary"]) < min_salary:
        min_salary = data_dict[k]["salary"]
    if float(data_dict[k]["salary"]) > max_salary:
        max_salary = data_dict[k]["salary"]
print(max_salary)
print(min_salary)  

print("Rescaled value of 200,000:", ((200000-min_salary)/(max_salary-min_salary)))

max_stock = float(-inf)
min_stock = float(inf)
for k in data_dict.keys():
    if float(data_dict[k]["exercised_stock_options"]) < min_stock:
        min_stock = data_dict[k]["exercised_stock_options"]
    if float(data_dict[k]["exercised_stock_options"]) > max_stock:
        max_stock = data_dict[k]["exercised_stock_options"]
print(max_stock)
print(min_stock)

print("Rescaled value of 1,000,000:", ((1000000-min_stock)/(max_stock-min_stock)))




### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn import cluster
cls = cluster.KMeans(n_clusters=2)
cls.fit(finance_features, poi)
pred = cls.predict(finance_features)



### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("No predictions object named pred found, no clusters to plot")
