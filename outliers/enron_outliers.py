#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
import numpy as np


sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
del data_dict['TOTAL']
# print "\n".join(data_dict['LAY KENNETH L'].keys())
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

print [a for a in data_dict.keys() if data_dict[a]["salary"] == 1111258]

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



