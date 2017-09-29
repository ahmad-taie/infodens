# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 12:53:28 2016

@author: admin
"""

import unittest
import importlib
import imp, os
import sys, inspect
from os import path
import difflib
import numpy as np




class Test_format(unittest.TestCase):
    
    
    def setUp(self):
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        fileName, pathname, description = imp.find_module('infodens')
        
        from infodens.preprocessor import preprocess
        self.prepObj = preprocess.Preprocess('testFile.txt')
        from infodens.formater import format
        
        from infodens.controller import controller
        self.conObj = controller.Controller('testconfig.txt')
        ch, ids, cids = self.conObj.loadConfig()
        
        from infodens.featurextractor import featureManager
        self.featMgrObj = featureManager.FeatureManager(4, self.conObj.featureIDs, self.conObj.featargs, self.prepObj, 1)
        
        self.conObj2 = controller.Controller('testconfig2.txt')
        self.conObj2.loadConfig()
        self.prepObj2 = preprocess.Preprocess('testFile.txt')
        self.featMgrObj2 = featureManager.FeatureManager(4, self.conObj2.featureIDs, self.conObj2.featargs, self.prepObj2, 1)
        
        self.features = self.featMgrObj2.callExtractors()
        self.prepObj3 = preprocess.Preprocess('labelFile.txt')
        self.labels = self.prepObj3.preprocessClassID()
        
        self.X = self.features
        self.y = np.asarray(self.labels)
        self.fmtObj = format.Format(self.X, self.y)
        
        
        
    def test_libsvmFormat(self):
        c = [[1, '1:2.5', '2:0.75'], [1, '1:3.25', '2:1.0'], [2, '1:2.75', '2:0.75'], [2, '1:3.5', '2:1.25']]
        ch =  self.fmtObj.libsvmFormat('libfmt.txt')
        self.assertListEqual(c,ch)
    
    def test_lexicalToTokens(self):
        c = [[2.5, 0.75, 1], [3.25, 1.0, 1], [2.75, 0.75, 2], [3.5, 1.25, 2]]
        ch = self.fmtObj.arrfFormat('arrffmt.txt')
        self.assertListEqual(c,ch)
        
   
        
    
        
    
        
if __name__ == '__main__':       
    unittest.main()