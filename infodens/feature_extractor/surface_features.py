# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 14:12:49 2016

@author: admin
"""
from infodens.feature_extractor.feature_extractor import featid, Feature_extractor
from scipy import sparse


class Surface_features(Feature_extractor):

    def getAverageWordLen(self, sentences):

        avgWordLen = []
        for sentence in sentences:
            if len(sentence) is 0:
                avgWordLen.append(0)
            else:
                length = sum([len(s) for s in sentence])
                avgWordLen.append(float(length) / len(sentence))

        return sparse.lil_matrix(avgWordLen).transpose()
    
    @featid(1)    
    def averageWordLength(self, argString, preprocessReq=0):

        '''Find average word length of every sentence and return list. '''

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            self.preprocessor.gettokenizeSents()
            self.testPreprocessor.gettokenizeSents()
            return 1

        aveWordLen = self.getAverageWordLen(self.preprocessor.gettokenizeSents())
        testAveWordLen = self.getAverageWordLen(self.testPreprocessor.gettokenizeSents())

        return aveWordLen, testAveWordLen, "Average Sentence's word length"

    def getSentLen(self, sentences):
        sentLen = []
        for sent in sentences:
            sentLen.append(len(sent))
        return sparse.lil_matrix(sentLen).transpose()

    @featid(10)
    def sentenceLength(self, argString, preprocessReq=0):
        '''Find length of every sentence and return list. '''

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            self.preprocessor.gettokenizeSents()
            self.testPreprocessor.gettokenizeSents()
            return 1

        sentLen = self.getSentLen(self.preprocessor.gettokenizeSents())
        testSentLen = self.getSentLen(self.testPreprocessor.gettokenizeSents())

        return sentLen, testSentLen, "Sentence Length"

    def getSyllableRatio(self, sentences, vowels):
        sylRatios = []
        for sent in sentences:
            sylCount = 0
            for word in sent:
                word2List = list(word)
                for i in range(len(word2List)-1):
                    if word2List[i] in vowels and word2List[i+1] not in vowels:
                        sylCount += 1

            if len(sent) is 0:
                sylRatios.append(0)
            else:
                sylRatios.append(float(sylCount)/len(sent))
        return sparse.lil_matrix(sylRatios).transpose()

    @featid(2)
    def syllableRatio(self, argString, preprocessReq=0):
        '''
        We approximate this feature by counting the number of vowel-sequences
        that are delimited by consonants or space in a word, normalized by the number of tokens
        in the chunk
        '''
        if preprocessReq:
            # Request all preprocessing functions to be prepared
            self.testPreprocessor.gettokenizeSents()
            self.preprocessor.gettokenizeSents()
            return 1

        if argString:
            vowels = argString.split(",")
        else:
            # Assume English vowels lowercase
            vowels = ['a', 'e', 'i', 'o', 'u']

        sylRatios = self.getSyllableRatio(self.preprocessor.gettokenizeSents(), vowels)
        testSylRatios = self.getSyllableRatio(self.testPreprocessor.gettokenizeSents(), vowels)

        return sylRatios, testSylRatios, "Syllable ratio"
