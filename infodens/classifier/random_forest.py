'''
Created on Aug 23, 2016

@author: admin
'''
from infodens.classifier.classifier import Classifier

import random
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


class Random_forest(Classifier):
    
    classifierName = 'Random Forest'
    n_estimators = 20
        
    def train(self):

        clf = RandomForestClassifier(n_estimators=20, n_jobs=self.threadCount)
        # Set the parameters by cross-validation
        # specify parameters and distributions to sample from
        param_dist = {"bootstrap": [True, False],
                          "criterion": ["gini", "entropy"]}

        n_iter_search = 4
        estimatorClass = RandomizedSearchCV(clf, param_distributions=param_dist,
                                  n_iter=n_iter_search, n_jobs=self.threadCount, cv=5)
        estimatorClass.fit(self.Xtrain, self.ytrain)

        self.model = estimatorClass.best_estimator_
