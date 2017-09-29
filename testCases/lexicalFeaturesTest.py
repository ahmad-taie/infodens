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




class Test_lexicalFeatures(unittest.TestCase):
    
    def setUp(self):
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        fileName, pathname, description = imp.find_module('infodens')        
        from infodens.preprocessor import preprocess
        self.prepObj = preprocess.Preprocess('testFile.txt')        
        from infodens.featurextractor import lexicalFeatures
        self.lexObj = lexicalFeatures.LexicalFeatures(self.prepObj)
        

    def test_lexicalDensity(self):
        c = [0.5, 0.25, 0.5, 0.25]
        ch = self.lexObj.lexicalDensity('NNP,NN,VBZ')
        self.assertListEqual(c,ch)
        
    def test_lexicalRichness(self):
        c = [1.0, 1.0, 1.0, 1.0]
        ch = self.lexObj.lexicalRichness('argString')
        self.assertListEqual(c,ch)
    
    def test_lexicalToTokens(self):
        c = [0.5, 1.0, 0.5, 1.0]
        ch = self.lexObj.lexicalToTokens('CC,DT,WDT,IN,PDT')
        self.assertListEqual(c,ch)
        
   
        
    
        
    
        
if __name__ == '__main__':    
    unittest.main()