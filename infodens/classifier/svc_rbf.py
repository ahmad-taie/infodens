'''
Created on Aug 23, 2016

@author: admin
'''

from infodens.classifier.classifier import Classifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np

import time


class SVC_rbf(Classifier):
    '''
    classdocs
    '''

    classifierName = 'SVC_rbf'
    C = np.logspace(-2, 10, 13)
    gamma = np.logspace(-9, 3, 13)

    def train(self):

        tuned_parameters = [{'kernel': ['rbf'], 'C': self.C, 'gamma':self.gamma}]

        print ('SVM Optimizing. This will take a while')
        start_time = time.time()
        clf = GridSearchCV(SVC(), tuned_parameters,
                           n_jobs=self.threadCount, cv=5)

        clf.fit(self.Xtrain, self.ytrain)
        print('Done with Optimizing. it took ', time.time() -
              start_time, ' seconds')

        self.model = clf.best_estimator_
