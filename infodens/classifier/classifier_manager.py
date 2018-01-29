# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 09:24:16 2016

@author: admin
"""
import importlib
import imp, os
import sys, inspect
from os import path
from infodens.classifier.classifier import Classifier


class Classifier_manager:

    def __init__(self, classifs, classifArgs, dSet, labs, threads=1, cv_folds=1, cv_Percent = 0,
                 persistFile="", persistOnFull=False):
        self.classifierArgs = classifArgs
        self.classifierIDs = classifs
        self.classifRank = []
        self.classifRankN = []
        self.persistFile = persistFile
        self.persistOnFull = persistOnFull
        self.dataSet = dSet
        self.labels = labs
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        self.fileName, self.pathname, self.description = imp.find_module('classifier')
        self.classifyModules = []
        self.cv_folds = cv_folds
        self.cv_Percent = cv_Percent
        self.threadsCount = threads
        self.availClassifiers = self.returnClassifiers()

    def checkParseClassifier(self):
        # Class_X r 100, Class_Y, Class_Z r

        # Check valid classifiers
        for classif in self.classifierIDs:
            if classif not in self.availClassifiers:
                # Not a valid classifier
                return 0

        for classifArg in self.classifierArgs:
            #print(classifArgs)
            if classifArg:
                classifArgs = classifArg.strip().split()
                if classifArgs[0] is "r":
                    self.classifRank.append(True)
                else:
                    print("Not a valid rank argument")
                    return 0
                if len(classifArgs) > 1:
                    if classifArgs[1].isdigit():
                        self.classifRankN.append(int(classifArgs[1]))
                    else:
                        print("Not a valid number of TopN features.")
                        return 0
                else:
                    # No topN, rank all feats
                    self.classifRankN.append(-1)
            else:
                # No required ranking for this classifier
                self.classifRank.append(False)
                self.classifRankN.append(-1)
        return 1

    def runClassifier(self, classifierToRun):
        classifReport = str(type(classifierToRun).__name__)
        classifReport += ":\n"
        classifReport += classifierToRun.runClassifier()
        classifReport += "\n"

        # Given file name to persist, persist classifier to file
        if self.persistFile:
            if self.persistOnFull:
                print("Training on full data to persist model..")
                classifierToRun.runClassifier(trainOnFull=True)
            classifierToRun.persist(self.persistFile)

        return classifReport

    def returnClassifiers(self):
        files = (os.listdir("infodens/classifier"))
        for file in files:
            if file.endswith(".py") and file is not "__init__.py":
                file = file.replace(".py",'')
                module = "infodens.classifier."+ file
                importlib.import_module(module)
                self.classifyModules.append(file)
        return [cls.__name__ for cls in Classifier.__subclasses__()]

    def callClassifiers(self):

        classifierObjs = []
        for classif in self.classifierIDs:
            for module in self.classifyModules:
                if classif.lower() == module.lower():
                    break
            classModule = importlib.import_module("infodens.classifier."+module)
            class_ = getattr(classModule, classif)
            classifierObjs.append(class_(self.dataSet, self.labels, self.threadsCount,
                                         self.cv_folds, self.cv_Percent))

        classifReports = []

        for i in range(0, len(classifierObjs)):
            classif = classifierObjs[i]
            report = self.runClassifier(classif)
            if self.classifRank[i]:
                print("Ranking features...")
                report += classif.rankFeats(self.classifRankN[i])
                print("Ranking done.")

            classifReports.append(report)

        return '\n'.join(classifReports)
