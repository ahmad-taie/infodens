'''
Created on Aug 23, 2016

@author: admin
'''

from infodens.classifier.classifier import Classifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import argparse

import time


class SVC_linear(Classifier):
    '''
    classdocs
    '''

    classifierName = 'SVC_linear'
    C = np.logspace(-5.0, 5.0, num=10, endpoint=True, base=2)

    def argParser(self):
        parser = argparse.ArgumentParser(description='{0} arguments.'.format(self.classifierName))
        parser.add_argument("-rank", help="Rank N features",
                            type=int, default=0)
        parser.add_argument("-cv", help="cross validation",
                            type=int, default=3)
        return parser.parse_args(self.args)

    def train(self):

        tuned_parameters = [{'C': self.C}]

        print('SVM Optimizing. This will take a while')
        start_time = time.time()
        clf = GridSearchCV(LinearSVC(verbose=0), tuned_parameters,
                           n_jobs=self.threadCount, cv=self.args.cv)

        clf.fit(self.Xtrain, self.ytrain)
        timeElaps = time.time() - start_time
        print("Done with Optimizing. It took {0:.2f} minutes.".format(timeElaps/60))

        self.model = clf.best_estimator_
