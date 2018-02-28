# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:32:17 2016

@author: admin
"""
from .format_writer import Format_writer
        

class Format:

    def __init__(self, fsX, fsy, featDescrips):
        self.X = fsX
        self.Y = fsy
        self.featDescriptions = featDescrips

    def outputDescriptor(self, fileName):
        with open(fileName, 'w') as classifOut:
            for i in range(0, len(self.featDescriptions)):
                classifOut.write("Feature Descriptors for run {0}:"
                                 " \r\n {1}".format(i, self.featDescriptions[i]))

    def outFormat(self, fileName, formatType):
        print("Writing features to file.")
        writer = Format_writer()
        writer.writeFormats(self.X, self.Y, fileName, formatType)
        self.outputDescriptor("feat_descriptors_{0}.desc".format(fileName))
        print("Feature file written.")


