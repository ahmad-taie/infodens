'''
Created on Aug 23, 2016

@author: admin
'''
from infodens.classifier.classifier import Classifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


class Ensemble(Classifier):

    classifierName = 'Ensemble'

    def train(self):

        #TODO: Define ensemble
        cl1 = RandomForestClassifier(random_state=1)
        listOfClassifiers = [("randomForest", cl1)]

        clf = VotingClassifier(estimators=listOfClassifiers, voting='hard')
        clf.fit(self.Xtrain, self.ytrain)
            
        self.model = clf
