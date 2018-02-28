'''
Created on Aug 23, 2016

@author: admin
'''

from infodens.classifier.classifier import Classifier
from sklearn.neural_network import MLPClassifier

class MLP_classifier(Classifier):
    '''
    classdocs
    '''

    classifierName = 'MLP_classifier'

    def train(self):

        clf = MLPClassifier()
        clf.fit(self.Xtrain, self.ytrain)
        self.model = clf
