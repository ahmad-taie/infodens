'''
Created on Aug 23, 2016

@author: admin
'''

from infodens.classifier.classifier import Classifier
from sklearn.neural_network import MLPClassifier
import argparse


class MLP_classifier(Classifier):
    '''
    classdocs
    '''

    classifierName = 'MLP_classifier'

    def argParser(self):
        parser = argparse.ArgumentParser(description='{0} arguments.'.format(self.classifierName))
        parser.add_argument("-rank", help="Rank N features",
                            type=int, default=0)
        parser.add_argument("-hidden_layers", help="Hidden layer sizes",
                            type=str, default="100")
        return parser.parse_args(self.args)

    def train(self):

        listOfHL = [int(layer) for layer in self.args.hidden_layers.split(",")]
        layers = tuple(listOfHL)
        print("Training {0} hidden layer(s) of size(s) {1}.".format(len(listOfHL),listOfHL))
        clf = MLPClassifier(hidden_layer_sizes=layers)
        clf.fit(self.Xtrain, self.ytrain)
        self.model = clf
