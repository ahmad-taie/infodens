'''
Created on Aug 23, 2016

@author: admin
'''

from infodens.classifier.classifier import Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import f1_score, recall_score, log_loss

import time


class Logistic_regression(Classifier):
    '''
    classdocs
    '''

    classifierName = 'Logistic_regression'

    def evaluate(self):
        y_pred = self.predict()
        return accuracy_score(self.ytest, y_pred),\
               precision_score(self.ytest, y_pred, average="weighted"),\
               recall_score(self.ytest, y_pred, average="weighted"),\
               f1_score(self.ytest, y_pred, average="weighted"),\
               log_loss(self.ytest, y_pred)


    def train(self):

        clf = LogisticRegression(n_jobs=self.threadCount)
        clf.fit(self.Xtrain, self.ytrain)
        self.model = clf

    def runClassifier(self, trainOnFull= False):
        """ Run the provided classifier."""
        acc = []; pre = []; rec = []; fsc = []; logloss = []

        if trainOnFull:
            self.Xtrain = self.X
            self.ytrain = self.y
            self.train()
            return "Trained on Full set."

        print("Cross validation on {:.1%} of data".format(self.cv_Percent))

        for i in range(self.n_foldCV):
            print("Training fold {0} of {1} for {2}..".format(i+1,
                                        self.n_foldCV, self.classifierName))
            self.shuffle()
            self.splitTrainTest()
            self.train()
            accu, prec, reca, fsco, lls = self.evaluate()
            acc.append(accu)
            pre.append(prec)
            rec.append(reca)
            fsc.append(fsco)
            logloss.append(lls)

        classifReport = 'Average Accuracy: ' + str(np.mean(acc))
        classifReport += '\nAverage Precision: ' + str(np.mean(pre))
        classifReport += '\nAverage Recall: ' + str(np.mean(rec))
        classifReport += '\nAverage F-score: ' + str(np.mean(fsc))
        classifReport += '\nAverage Log loss: ' + str(np.mean(logloss))

        return classifReport