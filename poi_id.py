#!/usr/bin/python

import sys
import pickle
#sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

import numpy as np
from matplotlib import pyplot as plt

features_list = ['poi','salary','bonus','shared_receipt_with_poi','from_this_person_to_poi','from_poi_to_this_person','expenses'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### #########    Task 2: Remove outliers  ################

    
data_temp = featureFormat(data_dict, features_list, sort_keys = True, remove_NaN = True)
salary = data_temp[:,1]
bonus = data_temp[:,2]
shared_receipt_with_poi = data_temp[:,3]
from_this_person_to_poi = data_temp[:,4]
from_poi_to_this_person = data_temp[:,5]
expenses = data_temp[:,6]
# plotting scatter plots to identify outliers

new_features_list = features_list[1:]
fig_count=0
for i in range(len(new_features_list)-2):
    i = i+2
    for j in range(i+1,len(new_features_list)):
        if i!=j:
            fig_count=fig_count+1
            plt.figure(fig_count)
            plt.plot(vars()[new_features_list[i]],vars()[new_features_list[j]],'.')
            plt.xlabel(new_features_list[i])
            plt.ylabel(new_features_list[j])

# plotting bonus vs salary
plt.figure()
plt.scatter(salary,bonus)
plt.xlabel('Salary')
plt.ylabel('bonus')


# removing the outlier ('TOTAL')
data_dict.pop('TOTAL',0)

data_new = featureFormat(data_dict, features_list, remove_NaN=True)  # data without outlier
salary_new=data_new[:,1]
bonus_new=data_new[:,2]
expenses_new=data_new[:,6]

# plotting salary vs bonus after removing the outlier 'TOTAL'
plt.figure()
plt.scatter(salary_new,bonus_new)
plt.xlabel('Salary')
plt.ylabel('bonus')


    
###########    Task 3: Create new feature(s)  ############


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



data_pts = len(data)
poi_pts = sum(data[:,0])
print('Number of data points: %i' % data_pts)
print('Number of POIs : %i' % poi_pts)
print('Number of features used : %i' % (len(features_list)-1))

feature_count=0
for f in features_list:
    print('Number of NaN in %s = %i' % (f,sum(np.isnan(data[:,feature_count]))))
    feature_count = feature_count+1


# removing NaN from data
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN=True)

#trying out a new feature

total_tofrom_poi = data_new[:,4] + data_new[:,5]
total_tofrom_poi = total_tofrom_poi.reshape(-1,1)
data = np.append(data,total_tofrom_poi,axis=1)

labels, features = targetFeatureSplit(data)


# implementing feature scaling values in some features are very big and varies in a larger range
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
features = scaler.fit_transform(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)



############ Trying out Gaussian NB ##############


clf_NB = GaussianNB()
clf_NB.fit(features_train,labels_train)
pred_NB = clf_NB.predict(features_test)
acc_NB = accuracy_score(pred_NB,labels_test)
print('Accuracy NB: %0.3f' % acc_NB)

precision_NB = precision_score(labels_test,pred_NB)
recall_NB = recall_score(labels_test,pred_NB)
print('Precision_NB = %0.3f' % precision_NB)
print('Recall_NB = %0.3f' % recall_NB)



####### Trying out Decision Tree Classifier ###################


param_grid_DT1 = {
             'min_samples_split': [4,8,2,6],
              'min_samples_leaf': [3,2,1,4],'max_depth':[2,3,4]}
clf_DT=GridSearchCV(tree.DecisionTreeClassifier(random_state = 0), param_grid_DT1)
clf_DT.fit(features_train,labels_train)
print "Best estimators for DT:"
print clf_DT.best_estimator_

pred_DT=clf_DT.predict(features_test)
acc_DT=accuracy_score(pred_DT, labels_test)
print('Accuracy DT = %0.3f' % acc_DT)

precision_DT = precision_score(labels_test,pred_DT)
recall_DT = recall_score(labels_test,pred_DT)
print('Precision_DT = %0.3f' % precision_DT)
print('Recall_DT = %0.3f' % recall_DT)





###### Trying out SVM ##########################

from sklearn.svm import SVC

param_grid_svm = {
             'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.001, 0.0005, 0.0001, 0.005, 0.01, 0.1],
              }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf_svm = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid_svm)
clf_svm = clf_svm.fit(features_train, labels_train)
print "Best estimators for SVM:"
print clf_svm.best_estimator_

pred_svm = clf_svm.predict(features_test)
acc_svm = accuracy_score(pred_svm, labels_test)
print('Accuracy SVM = %0.3f' % acc_svm)


precision_svm = precision_score(labels_test,pred_svm)
recall_svm = recall_score(labels_test,pred_svm)
print('Precision_svm = %0.3f' % precision_svm)
print('Recall_svm = %0.3f' % recall_svm)


###########################################


### getting the importance values for each feature using DT

important_features=clf_DT.best_estimator_.feature_importances_
new_features_list = new_features_list + ['total_tofrom_poi']
idx_features = 0
for idx in important_features:
    print idx
    if (idx > 0.02):
        print('Feature = %s' % new_features_list[idx_features], 'Importance = %0.5f' % idx )      # printing features with importance > 0.2
    idx_features = idx_features + 1
        
max_value=max(important_features)               # max importance value
index_max = np.argmax(important_features)       # index of max. importance value

print('Max. feature importance value = %0.7f' % max_value, 'Index of most important feature = %i' % index_max)
print('Most important feature = %s' % new_features_list[index_max])




############  Final Analysis ###########

# NOT including the new feature

# Classifier used = Decision Trees (DT)

my_dataset = data_dict

# select the three most important features (from feature importances)
features_list = ['poi','shared_receipt_with_poi','from_this_person_to_poi','expenses']

data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN=True)
labels, features = targetFeatureSplit(data)

# implementing feature scaling values in some features are very big and varies in a larger range
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
features_scaled = scaler.fit_transform(features)


features_train, features_test, labels_train, labels_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)


param_grid_DT = {
             'min_samples_split': [2,4,8,6],
              'min_samples_leaf': [1,2,3,4]}
clf=GridSearchCV(tree.DecisionTreeClassifier(random_state = 42), param_grid_DT)
#clf=tree.DecisionTreeClassifier(random_state = 42, min_samples_leaf=2, min_samples_split=2)
clf.fit(features_train,labels_train)
print "Best estimators for DT:"
print clf.best_estimator_

pred=clf.predict(features_test)
acc=accuracy_score(pred, labels_test)
print('Accuracy DT = %0.3f' % acc)

precision = precision_score(labels_test,pred)
recall = recall_score(labels_test,pred)
print('Precision_DT = %0.3f' % precision)
print('Recall_DT = %0.3f' % recall)


#### Other algorithm tried in the final analysis

#####################################


### Naive Bayes

clf_NB = GaussianNB()
clf_NB.fit(features_train,labels_train)
pred_NB = clf_NB.predict(features_test)
acc_NB = accuracy_score(pred_NB,labels_test)
print('Accuracy NB: %0.3f' % acc_NB)

precision_NB = precision_score(labels_test,pred_NB)
recall_NB = recall_score(labels_test,pred_NB)
print('Precision_NB = %0.3f' % precision_NB)
print('Recall_NB = %0.3f' % recall_NB)


################################

# SVM

param_grid_svm = {
             'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.001, 0.0005, 0.0001, 0.005, 0.01, 0.1],
              }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf_svm = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid_svm)
clf_svm = clf_svm.fit(features_train, labels_train)
print "Best estimators for SVM:"
print clf_svm.best_estimator_

pred_svm = clf_svm.predict(features_test)
acc_svm = accuracy_score(pred_svm, labels_test)
print('Accuracy SVM = %0.3f' % acc_svm)


precision_svm = precision_score(labels_test,pred_svm)
recall_svm = recall_score(labels_test,pred_svm)
print('Precision_svm = %0.3f' % precision_svm)
print('Recall_svm = %0.3f' % recall_svm)

#################################



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
#plt.show()

