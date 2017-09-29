# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:44:50 2016

@author: admin
"""

import unittest
import os, imp
import sys
from os import path


if __name__ == '__main__':
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
    fileName, pathname, description = imp.find_module('testCases')
    
    from testCases import bagOfNgramsTest, classifierManagerTest, controllerTest, featureManagerTest, formatTest, lexicalFeaturesTest, preprocessorTest, surfaceFeaturesTest
    
    
    
    surfFeatTests = surfaceFeaturesTest.Test_surfaceFeatures
    bagNgTests = bagOfNgramsTest.Test_bagOfNgrams
    clfMgrTest = classifierManagerTest.Test_classifierManager
    ctrlTest = controllerTest.Test_controller
    ftMgrTest = featureManagerTest.Test_featureManager
    fmtTest = formatTest.Test_format
    lexFtTest = lexicalFeaturesTest.Test_lexicalFeatures
    prepTest = preprocessorTest.Test_preprocess
    
    
    
    test_classes_to_run = [surfFeatTests, bagNgTests, clfMgrTest, ctrlTest,  ftMgrTest, fmtTest, lexFtTest, prepTest]
    
    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)