# %%
# DO THIS TO INCLUDE UTILS
import os, sys
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)
# 

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from torchmetrics.classification import BinaryFBetaScore
from sklearn.model_selection import train_test_split

from datetime  import datetime
from tqdm      import tqdm

from utils import process_data, output_csv

import numpy as np

# %%
def classification_err(y, real_y):
    """
    This function returns the classification error between two equally-sized vectors of 
    labels; this is the fraction of samples for which the labels differ.
    
    Inputs:
        y: (N, ) shaped array of predicted labels
        real_y: (N, ) shaped array of true labels
    Output:
        Scalar classification error
    """
    # tot_diff = 0
    # for i in range(y.shape[0]):
    #     if y[i] != real_y[i]:
    #         tot_diff += 1

    # print(tot_diff)
    # print(np.count_nonzero(y != real_y))
    # turn tot_diff/y.shape[0]
    return np.count_nonzero(y != real_y)/y.shape[0]

def eval_tree_based_model_max_depth(clf, max_depth, X_train, y_train, X_test, y_test):
    """
    This function evaluates the given classifier (either a decision tree or random forest) at all of the 
    maximum tree depth parameters in the vector max_depth, using the given training and testing
    data. It returns two vector, with the training and testing classification errors.
    
    Inputs:
        clf: either a decision tree or random forest classifier object
        max_depth: a (T, ) vector of all the max_depth stopping condition parameters 
                            to test, where T is the number of parameters to test
        X_train: (N, D) matrix of training samples.
        y_train: (N, ) vector of training labels.
        X_test: (N, D) matrix of test samples
        y_test: (N, ) vector of test labels
    Output:
        train_err: (T, ) vector of classification errors on the training data
        test_err: (T, ) vector of classification errors on the test data
    """
    train_err = []
    test_err = []
    
    for max_d in max_depth:
        clf.set_params(max_depth=max_d)
        clf.fit(X_train, y_train)
        # tree.plot_tree(clf)

        train_err.append(classification_err(clf.predict(X_train), y_train))
        test_err.append(classification_err(clf.predict(X_test), y_test))

    return train_err, test_err

# %% 

TRAIN_BASIC_FEATURES_PATH = '../data/train_7_feat.csv'
TEST_BASIC_FEATURES_PATH = '../data/test_7_feat.csv'

# %%

n_estimators = 1000
clf = RandomForestClassifier(n_estimators = n_estimators, criterion = 'gini')

max_depth = np.arange(2, 21)

train_err, test_err = eval_tree_based_model_max_depth(clf, max_depth, train_data, 
                                                        train_label, test_data, test_label)