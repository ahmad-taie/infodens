'''
Created on Aug 23, 2016

@author: admin
'''
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score, precision_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score
import argparse


class Classifier(object):
    '''
    classdocs
    '''

    Xtrain = []
    ytrain = []
       
    Xtest = []
    ytest = []

    threadCount = 1

    classifierName = ''
    
    def __init__(self, dataX, datay, testX, testY, args, threads=1):
        self.Xtrain = dataX
        self.ytrain = datay
        self.Xtest = testX
        self.ytest = testY
        self.threadCount = threads
        self.rankReport = ""
        self.model = None
        print("Using {0} with arguments : {1}".format(self.classifierName, args))
        if args:
            self.args = args.split()
        else:
            self.args = ""
        self.args = self.argParser()
        self.train()

    def argParser(self):
        parser = argparse.ArgumentParser(description='{0} arguments.'.format(self.classifierName))
        parser.add_argument("-rank", help="Rank N features",
                            type=int, default=0)
        return parser.parse_args(self.args)

    def predict(self):
        print("Predicting labels for {0}.".format(self.classifierName))
        return self.model.predict(self.Xtest)

    def persist(self, fileName):
        import joblib
        output = "{0}_{1}".format(self.classifierName, fileName)

        joblib.dump(self.model, output)

    def evaluate(self):
        print("Evaluating test set..")
        y_pred = self.predict()
        return accuracy_score(self.ytest, y_pred),\
               precision_score(self.ytest, y_pred, average="weighted"),\
               recall_score(self.ytest, y_pred, average="weighted"),\
               f1_score(self.ytest, y_pred, average="weighted")

    def rankFeats(self):
        rankN = self.args.rank
        if not rankN:
            return ""

        print("Ranking  features...")
        # Override for regression and classifiers with readily available
        # Rankers
        from sklearn.feature_selection import mutual_info_classif
        ranking = mutual_info_classif(self.Xtrain, self.ytrain)

        outStr = "Ordered Mutual information and feature index:\n"

        self.rankReport = outStr + str(sorted(enumerate(ranking),
                                              key=lambda x: x[1], reverse=True)[:rankN])

        return self.rankReport

    def evaluateClassifier(self):
        """ Run the provided classifier."""

        # Test classes provided, score...
        accu, prec, reca, fsco = self.evaluate()
        classifReport = 'Accuracy: ' + str(accu)
        classifReport += '\nPrecision: ' + str(prec)
        classifReport += '\nRecall: ' + str(reca)
        classifReport += '\nF-score: ' + str(fsco)
        return classifReport


