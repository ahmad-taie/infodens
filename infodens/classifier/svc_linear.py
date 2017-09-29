'''
Created on Aug 23, 2016

@author: admin
'''

from infodens.classifier.classifier import Classifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import numpy as np

import time


class SVC_linear(Classifier):
    '''
    classdocs
    '''

    classifierName = 'Support Vector Machine'
    C = np.logspace(-5.0, 5.0, num=10, endpoint=True, base=2)

    def train(self):

        tuned_parameters = [{'C': self.C}]

        print ('SVM Optimizing. This will take a while')
        start_time = time.time()
        clf = GridSearchCV(LinearSVC(), tuned_parameters,
                           n_jobs=self.threadCount, cv=5)

        clf.fit(self.Xtrain, self.ytrain)
        print('Done with Optimizing. it took ', time.time() -
              start_time, ' seconds')

        self.model = clf.best_estimator_
