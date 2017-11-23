'''
Created on Aug 23, 2016

@author: admin
'''
import scipy
import numpy as np
import sklearn
import random
from sklearn import model_selection
import os
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score


class Classifier(object):
    '''
    classdocs
    '''
        
    X = []
    y = []

    Xtrain = []
    ytrain = []
       
    Xtest = []
    ytest = []
       
    default_splitPercent = 0.3
    n_foldCV = 0
    threadCount = 1
    model = 0

    classifierName = ''
    
    def __init__(self, dataX, datay, threads=1, nCrossValid=2, cv_Percent=0):
        self.X = dataX
        self.y = datay
        self.threadCount = threads
        self.n_foldCV = nCrossValid
        if cv_Percent:
            self.cv_Percent = cv_Percent
        else:
            self.cv_Percent = self.default_splitPercent

    def shuffle(self):
        self.X, self.y = sklearn.utils.shuffle(self.X, self.y)
        
    def splitTrainTest(self):
        #print(self.cv_Percent)
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = \
            model_selection.train_test_split(self.X, self.y,
                                             test_size=self.cv_Percent, random_state=0)

    def predict(self):
        return self.model.predict(self.Xtest)

    def persist(self, fileName):
        import joblib
        output = "{0}_{1}".format(self.classifierName, fileName)

        joblib.dump(self.model, output)

    def evaluate(self):
        y_pred = self.predict()
        return accuracy_score(self.ytest, y_pred),\
               precision_score(self.ytest, y_pred, average="weighted"),\
               recall_score(self.ytest, y_pred, average="weighted"),\
               f1_score(self.ytest, y_pred, average="weighted")

    def rankFeats(self, rankN=-1):
        # Override for regression and classifiers with readily available
        # Rankers
        from sklearn.feature_selection import mutual_info_classif
        ranking = mutual_info_classif(self.X, self.y)

        outStr = "Ordered Mutual information and feature index:\n"

        self.rankReport = outStr + str(sorted(enumerate(ranking),
                                              key=lambda x: x[1], reverse=True)[:rankN])

        return self.rankReport

    def runClassifier(self, trainOnFull= False):
        """ Run the provided classifier."""
        acc = []; pre = []; rec = []; fsc = []

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
            accu, prec, reca, fsco = self.evaluate()
            acc.append(accu)
            pre.append(prec)
            rec.append(reca)
            fsc.append(fsco)

        classifReport = 'Average Accuracy: ' + str(np.mean(acc))
        classifReport += '\nAverage Precision: ' + str(np.mean(pre))
        classifReport += '\nAverage Recall: ' + str(np.mean(rec))
        classifReport += '\nAverage F-score: ' + str(np.mean(fsc))

        return classifReport