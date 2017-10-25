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

    def __init__(self, ids, dSet, labs, threads=1, cv_folds=1):
        self.classifierArgs = ids
        self.classifierIDs = []
        self.classifRank = []
        self.classifRankN = []
        self.dataSet = dSet
        self.labels = labs
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        self.fileName, self.pathname, self.description = imp.find_module('classifier')
        self.classifyModules = []
        self.cv_folds = cv_folds
        self.threadsCount = threads
        self.availClassifiers = self.returnClassifiers()

    def checkParseClassifier(self):
        # Class_X r 100, Class_Y, Class_Z r
        #print(self.availClassifiers)
        for classifArg in self.classifierArgs:
            classifArgs = classifArg.strip().split()
            #print(classifArgs)
            if classifArgs[0] in self.availClassifiers:
                self.classifierIDs.append(classifArgs[0])
                if len(classifArgs) > 1:
                    if classifArgs[1] is "r":
                        self.classifRank.append(True)
                    else:
                        print("Not a valid rank argument")
                        return 0
                    if len(classifArgs) > 2:
                        if classifArgs[2].isdigit():
                            self.classifRankN.append(int(classifArgs[2]))
                        else:
                            print("Not a valid number of TopN features.")
                            return 0
                    else:
                        self.classifRankN.append(-1)
                else:
                    # No required ranking for this classifier
                    self.classifRank.append(False)
                    self.classifRankN.append(-1)
            else:
                # Not a valid classifier
                return 0
        return 1

    def runClassifier(self, classifierToRun):
        classifReport = str(type(classifierToRun).__name__)
        classifReport += ":\n"
        classifReport += classifierToRun.runClassifier()
        classifReport += "\n"
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
            classifierObjs.append(class_(self.dataSet, self.labels, self.threadsCount, self.cv_folds))

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
