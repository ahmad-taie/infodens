
import unittest
import importlib
import imp, os
import sys, inspect
from os import path
import difflib






class Test_featureManager(unittest.TestCase):
    
    def setUp(self):
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        fileName, pathname, description = imp.find_module('infodens')
        from infodens.preprocessor import preprocess
        self.prepObj = preprocess.Preprocess('testFile.txt')        
        
        from infodens.controller import controller
        self.conObj = controller.Controller('testconfig.txt')
        ch, ids, cids = self.conObj.loadConfig()
        
        from infodens.featurextractor import featureManager
        self.featMgrObj = featureManager.FeatureManager(4, self.conObj.featureIDs, self.conObj.featargs,self. prepObj, 1)
        
        self.conObj2 = controller.Controller('testconfig2.txt')
        self.conObj2.loadConfig()
        self.prepObj2 = preprocess.Preprocess('testFile.txt')
        self.featMgrObj2 = featureManager.FeatureManager(4, self.conObj2.featureIDs, self.conObj2.featargs, self.prepObj2, 1)

    def test_idClassDictionary(self):        
        chALlIds = {1: 'averageWordLength', 2: 'syllableRatio', 3: 'lexicalDensity', 4: 'ngramBagOfWords', 5: 'ngramPOSBagOfWords', 6: 'ngramMixedBagOfWords', 7: 'ngramLemmaBagOfWords', 8: 'parseTreeDepth', 10: 'sentenceLength', 11: 'lexicalRichness', 12: 'lexicalToTokens'}
        ids, allIds = self.featMgrObj.idClassDictionary()
        self.assertDictEqual(chALlIds,allIds)
        
    def test_methodsWithDecorator(self):        
        chmwDec = {8: 'parseTreeDepth', 1: 'averageWordLength', 10: 'sentenceLength', 2: 'syllableRatio'}
        idCMs, allIds = self.featMgrObj.idClassDictionary()
        mwDec = self.featMgrObj.methodsWithDecorator(idCMs[1], 'featid')
        self.assertDictEqual(chmwDec,mwDec)
        
    def test_checkFeatValidity(self):        
        c = 1
        idCMs, allIds = self.featMgrObj.idClassDictionary()
        ch = self.featMgrObj.checkFeatValidity()
        self.assertEquals(c,ch)
        
    def test_callExtractors(self):        
        c = [[2.5, 3.25, 2.75, 3.5], [0.75, 1.0, 0.75, 1.25]]        
        chArr = self.featMgrObj2.callExtractors()
        ch = chArr.tolist()
        self.assertListEqual(c,ch)
        
    
        
    
        
    
        
    
        
    
        
if __name__ == '__main__':
    unittest.main()