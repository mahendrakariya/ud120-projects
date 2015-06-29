#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
#################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

clf = GaussianNB()
clf.fit(features_train, labels_train)
print 'GaussianNB:', clf.score(features_test, labels_test)

clf = SVC()
clf.fit(features_train, labels_train)
print 'SVC:', clf.score(features_test, labels_test)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print 'DecisionTreeClassifier:', clf.score(features_test, labels_test)

clf = AdaBoostClassifier(n_estimators=100)
clf.fit(features_train, labels_train)
print 'AdaBoostClassifier:', clf.score(features_test, labels_test)

clf = KNeighborsClassifier(n_neighbors=10, weights='distance')
clf.fit(features_train, labels_train)
print 'KNeighborsClassifier:', clf.score(features_test, labels_test)

clf = RandomForestClassifier(n_estimators=1000, criterion='entropy')
clf.fit(features_train, labels_train)
print 'RandomForestClassifier:', clf.score(features_test, labels_test)

# try:
#     # prettyPicture(clf, features_test, labels_test)
# except NameError:
#     pass
