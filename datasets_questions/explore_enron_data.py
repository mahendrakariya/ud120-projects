#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
# print enron_data.keys()
# print enron_data['METTS MARK']
# salaries_null = [p for p in enron_data.keys() if enron_data[p]['total_payments'] == 'NaN' and enron_data[p]['poi'] == True]
# print salaries_null
print float(len(enron_data.keys()))



