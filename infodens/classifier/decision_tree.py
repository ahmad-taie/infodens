'''
Created on Aug 23, 2016

@author: admin
'''
from infodens.classifier.classifier import Classifier
import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree


class Decision_tree(Classifier):

    classifierName = 'Decision Tree'
    n_estimators = 20

    def train(self):

        estimatorClass = tree.DecisionTreeClassifier()
        estimatorClass.fit(self.Xtrain, self.ytrain)
        self.model = estimatorClass
