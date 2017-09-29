# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:47:06 2016

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:47:06 2016

@author: admin
"""

import unittest
import importlib
import imp, os
import sys, inspect
from os import path
import difflib
import time

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
fileName, pathname, description = imp.find_module('infodens')
from infodens.preprocessor import preprocess
prepObj = preprocess.Preprocess('testFile.txt')


class Test_preprocess(unittest.TestCase):
    
    def setUp(self):
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        fileName, pathname, description = imp.find_module('infodens')
        from infodens.preprocessor import preprocess
        self.prepObj = preprocess.Preprocess('testFile.txt')
        

    def test_preprocessBySentence(self):
        s = ['This is a boy', 'His name is Audu', 'This is a girl', 'Her name is Sarah']
        ps = self.prepObj.preprocessBySentence()
        self.assertListEqual(s,ps)
        
    def test_getPlainSentences(self):
        s = ['This is a boy', 'His name is Audu', 'This is a girl', 'Her name is Sarah']
        ps = self.prepObj.getPlainSentences()
        self.assertListEqual(s,ps)
        
    def test_gettokenizeSents(self):
        s = [['This', 'is', 'a', 'boy'], ['His', 'name', 'is', 'Audu'], ['This', 'is', 'a', 'girl'], ['Her', 'name', 'is', 'Sarah']]
        ps = self.prepObj.gettokenizeSents()
        self.assertListEqual(s,ps)
        
    def test_nltkPOStag(self):
        s = [['DT', 'VBZ', 'DT', 'NN'], ['PRP$', 'NN', 'VBZ', 'NNP'], ['DT', 'VBZ', 'DT', 'NN'], ['PRP$', 'NN', 'VBZ', 'NNP']]
        ps = self.prepObj.nltkPOStag()
        self.assertListEqual(s,ps)
        
    def test_getLemmatizedSents(self):
        s = [['This', 'is', 'a', 'boy'], ['His', 'name', 'is', 'Audu'], ['This', 'is', 'a', 'girl'], ['Her', 'name', 'is', 'Sarah']]
        ps = self.prepObj.getLemmatizedSents()
        self.assertListEqual(s,ps)
        
    def test_getMixedSents(self):
        s = [['This', 'VBZ', 'a', 'NN'], ['His', 'NN', 'VBZ', 'NNP'], ['This', 'VBZ', 'a', 'NN'], ['Her', 'NN', 'VBZ', 'NNP']]
        ps = self.prepObj.getMixedSents()
        self.assertListEqual(s,ps)
        
    
    def test_buildTokenNgrams(self):
        s = {('a', 'boy'): 1, ('name', 'is'): 2, ('a', 'girl'): 1, ('This', 'is'): 2, ('is', 'a'): 2, ('is', 'Sarah'): 1, ('is', 'Audu'): 1, ('Her', 'name'): 1, ('His', 'name'): 1}
        ps = self.prepObj.buildTokenNgrams(2)
        self.assertDictEqual(s, ps)
        
    def test_buildPOSNgrams(self):
        s = {('VBZ', 'DT'): 2, ('NN', 'VBZ'): 2, ('PRP$', 'NN'): 2, ('VBZ', 'NNP'): 2, ('DT', 'NN'): 2, ('DT', 'VBZ'): 2}
        ps = self.prepObj.buildPOSNgrams(2)
        self.assertDictEqual(s, ps)
        
    def test_buildLemmaNgrams(self):
        s = {('a', 'boy'): 1, ('name', 'is'): 2, ('a', 'girl'): 1, ('This', 'is'): 2, ('is', 'a'): 2, ('is', 'Sarah'): 1, ('is', 'Audu'): 1, ('Her', 'name'): 1, ('His', 'name'): 1}
        ps = self.prepObj.buildLemmaNgrams(2)
        self.assertDictEqual(s, ps) 
        
    def test_buildMixedNgrams(self):
        s = {('This', 'VBZ'): 2, ('a', 'NN'): 2, ('NN', 'VBZ'): 2, ('His', 'NN'): 1, ('VBZ', 'NNP'): 2, ('Her', 'NN'): 1, ('VBZ', 'a'): 2}
        ps = self.prepObj.buildMixedNgrams(2)
        self.assertDictEqual(s, ps) 
        
    def test_ngramMinFreq(self):
        s = {('This', 'VBZ'): 0, ('a', 'NN'): 3, ('VBZ', 'a'): 2, ('NN', 'VBZ'): 4, ('VBZ', 'NNP'): 1}
        ss = {('This', 'VBZ'): 2, ('a', 'NN'): 2, ('NN', 'VBZ'): 2, ('His', 'NN'): 1, ('VBZ', 'NNP'): 2, ('Her', 'NN'): 1, ('VBZ', 'a'): 2}
        ps, ind = self.prepObj.ngramMinFreq(ss, 2)
        self.assertDictEqual(s, ps) 
        
if __name__ == '__main__':
    unittest.main()
    




