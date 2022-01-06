#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib
from numpy.core.numeric import NaN

enron_data = joblib.load(open("./ud120projects/final_project/final_project_dataset.pkl", "rb"))
print(len(enron_data))
print(len(enron_data[list(enron_data.keys())[0]]))

counter = 0
email_counter = 0
salary_counter = 0
no_total_pay = 0
no_total_pay_pois = 0
for key in enron_data.keys():
    if enron_data[key]["poi"] == 1:
        counter += 1
    if enron_data[key]["email_address"] != "NaN":
        email_counter += 1
    if enron_data[key]["salary"] != "NaN":
        salary_counter += 1
    if enron_data[key]["total_payments"] == "NaN":
        no_total_pay += 1
    if (enron_data[key]["total_payments"] == "NaN") and (enron_data[key]["poi"] == 1):
        no_total_pay_pois += 1

print("Counter:", counter, "\nEmail:", email_counter, "\nSalary:", salary_counter, "\nNo Total Payment:", no_total_pay, "\nNo Total Payment for a POI:", no_total_pay_pois)

"""What you'll find here is that all POI's have total payment information, so if a machine learning algorithm
were to use total_payments as a feature, it would associate a NaN value with Non POI's.
In other words, no training points would have NaN when the class label is POI. 
If you add 10 POI's with NaN for total payments, the algorithm may interpret “NaN” for total_payments as a clue that someone is a POI

Adding in the new POI’s in this example, none of whom we have financial information for, has introduced a subtle problem, 
that our lack of financial information about them can be picked up by an algorithm as a clue that they’re POIs. 
Another way to think about this is that there’s now a difference in how we generated the data for our two classes-
-non-POIs all come from the financial spreadsheet, while many POIs get added in by hand afterwards. 
That difference can trick us into thinking we have better performance than we do-
-suppose you use your POI detector to decide whether a new, unseen person is a POI, and that person isn’t on the spreadsheet. 
Then all their financial data would contain “NaN” but the person is very likely not a POI (there are many more non-POIs than POIs in the world, 
and even at Enron)--you’d be likely to accidentally identify them as a POI, though!

This goes to say that, when generating or augmenting a dataset, 
you should be exceptionally careful if your data are coming from different sources for different classes. 
It can easily lead to the type of bias or mistake that we showed here. 
There are ways to deal with this, for example, you wouldn’t have to worry about this problem if you used only email data--in that case, 
discrepancies in the financial data wouldn’t matter because financial features aren’t being used. 
There are also more sophisticated ways of estimating how much of an effect these biases can have on your final answer; 
those are beyond the scope of this course.

For now, the takeaway message is to be very careful about introducing features that come from different sources depending on the class! 
It’s a classic way to accidentally introduce biases and mistakes."""


#print(list(enron_data.values())[0])
print(enron_data["PRENTICE JAMES"]["total_stock_value"])
print(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])
print(enron_data["SKILLING JEFFREY K"]['total_payments'])
print(enron_data["LAY KENNETH L"]['total_payments'])
print(enron_data["FASTOW ANDREW S"]['total_payments'])


