# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 14:42:43 2016

@author: admin
"""
import inspect
from infodens.preprocessor.preprocess_services import Preprocess_Services


def featid(func_id):
    def decorator(f):
        f.__name__ = "{0}_featid_{1}".format(f.__name__, func_id)
        return f
    return decorator


class Feature_extractor(object):

    def __init__(self, preprocessed):
        '''
        Initializes the class with a preprocessor. '''
        self.preprocessor = preprocessed
        self.prep_servs = Preprocess_Services()

