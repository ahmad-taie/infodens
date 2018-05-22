'''
Created on Aug 23, 2016

@author: admin
'''

from infodens.classifier.classifier import Classifier
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
import numpy


# Function to create model, required for KerasClassifier
def create_model(inputSize=None, layers=[10], n_classes=None, optimizer='adam', dropoutRate=0.3,
                 init='glorot_uniform'):
    # create model

    def modelCreator():
        model = Sequential()
        model.add(Dense(layers[0], input_dim=inputSize, kernel_initializer=init, activation='relu'))
        model.add(Dropout(dropoutRate))
        for i in range(1, len(layers)):
            model.add(Dense(layers[i], activation='relu'))
            model.add(Dropout(dropoutRate))
        model.add(Dense(n_classes, kernel_initializer=init, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    return modelCreator


class Keras_MLP(Classifier):
    '''
    classdocs
    '''

    classifierName = 'Keras_MLP'

    def argParser(self):
        parser = argparse.ArgumentParser(description='{0} arguments.'.format(self.classifierName))
        parser.add_argument("-rank", help="Rank N features",
                            type=int, default=0)
        parser.add_argument("-hidden_layers", help="Hidden layer sizes",
                            type=str, default="100")
        parser.add_argument("-epochs", help="Epochs to train for",
                            type=int, default=5)
        parser.add_argument("-batch_size", help="Size of mini batch",
                            type=int, default=64)
        parser.add_argument("-dropout", help="Dropout rate",
                            type=float, default=0.3)
        return parser.parse_args(self.args)

    def train(self):

        listOfHL = [int(layer) for layer in self.args.hidden_layers.split(",")]
        print("Training {0} hidden layer(s) of size(s) {1}.".format(len(listOfHL), listOfHL))
        print("Using dropout with rate: {0}".format(self.args.dropout))
        inputDim = self.Xtrain.get_shape()[1]

        n_classes = len(set(self.ytrain))

        clf = KerasClassifier(build_fn=create_model(inputDim, listOfHL, n_classes,
                                                    dropoutRate=self.args.dropout),
                              epochs=self.args.epochs, batch_size=64, verbose=1)
        clf.fit(self.Xtrain, self.ytrain)
        self.model = clf
