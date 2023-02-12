#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import make_scorer, roc_auc_score
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


# In[2]:


#Import scikit-learn dataset library
from sklearn import datasets

#Load dataset
cancer = datasets.load_breast_cancer()

print(cancer.target_names)
print(cancer.feature_names)


# In[3]:


data_y = cancer.target
data_x = cancer.data
# print the names of the 13? 30 features
print(data_x)

# print the label type of cancer('malignant' 'benign')
print(data_y)


# In[4]:


def split_data(data_x, data_y, n_split):
    kf = StratifiedKFold(n_splits=n_split)
    tr_x = []
    tr_y = []
    te_x = []
    te_y = []
    for train_idx, test_idx in kf.split(data_x, data_y):
        train_x_cv = data_x[train_idx, :]
        train_y_cv = data_y[train_idx]
        test_x = data_x[test_idx, :]
        test_y = data_y[test_idx]

        tr_x.append(train_x_cv)
        tr_y.append(train_y_cv)
        te_x.append(test_x)
        te_y.append(test_y)

    return tr_x, tr_y, te_x, te_y


def calc_specificity(y_actual, y_pred, thresh=0.5):
    # calculates specificity
    return sum((y_pred < thresh) & (y_actual == 0)) / sum(y_actual == 0)


def print_report(y_actual, y_pred, thresh):
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    specificity = calc_specificity(y_actual, y_pred, thresh)
    print('AUC:%.3f' % auc)
    print('accuracy:%.3f' % accuracy)
    print('recall:%.3f' % recall)
    print('precision:%.3f' % precision)
    print('specificity:%.3f' % specificity)
    print(' ')
    return auc, accuracy, recall, precision, specificity


def report_performance(clf, train_x, train_y, test_x, test_y, thresh=0.5, clf_name="CLF"):
    print("[x] performance for {} classifier".format(clf_name))
    y_train_preds = clf.predict_proba(train_x)[:, 1]
    y_test_preds = clf.predict_proba(test_x)[:, 1]
    print('Training:')
    train_auc, train_accuracy, train_recall, train_precision, train_specificity = print_report(train_y, y_train_preds,
                                                                                               thresh)
    print('Test:')
    test_auc, test_accuracy, test_recall, test_precision, test_specificity = print_report(test_y, y_test_preds, thresh)
    return {"train": {"auc": train_auc, "acc": train_accuracy, "recall": train_recall, "precision": train_precision,
                      "specificity": train_specificity},
            "test": {"auc": test_auc, "acc": test_accuracy, "recall": test_recall, "precision": test_precision,
                     "specificity": test_specificity}}


# In[5]:


data_x.shape


# In[6]:


data_y


# In[7]:


tr_x, tr_y, te_x, te_y = split_data(data_x, data_y, n_split=5)


# In[8]:


te_x[0].shape


# In[9]:


print(len(tr_x))
print(tr_x[0].shape)


# # Decision Tree

# In[10]:


dtree = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=5)
dtree.fit(tr_x[0], tr_y[0])
dtree.predict(te_x[0])
roc_auc_score(te_y[0], dtree.predict(te_x[0]))


# # Grid Search

# In[11]:


parameters = {"criterion": ["gini", "entropy"],
                  "max_depth": [3, 5, 10],
                  "min_samples_split": [5, 10, 15]}
n_split = 5
auc_scoring = make_scorer(roc_auc_score)
grid_clf = GridSearchCV(estimator=dtree, param_grid=parameters, cv=n_split, scoring=auc_scoring, verbose=0)
grid_clf.fit(tr_x[0], tr_y[0])


# In[12]:


print(grid_clf.best_estimator_)
print(grid_clf.best_params_)

report_performance(grid_clf.best_estimator_, tr_x[0], tr_y[0], te_x[0], te_y[0], clf_name="DecisionTree")

