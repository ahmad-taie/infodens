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
        self.classifierIDs = ids
        self.dataSet = dSet
        self.labels = labs
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        self.fileName, self.pathname, self.description = imp.find_module('classifier')
        self.classifyModules = []
        self.cv_folds = cv_folds
        self.threadsCount = threads
        self.returnClassifiers()

    def checkValidClassifier(self):
        #print(self.availClassifiers)
        for classifID in self.classifierIDs:
            if classifID not in self.availClassifiers:
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
        self.availClassifiers = [cls.__name__ for cls in Classifier.__subclasses__()]


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
        for classif in classifierObjs:
            classifReports.append(self.runClassifier(classif))

        return '\n'.join(classifReports)
