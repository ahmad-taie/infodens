# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 14:12:49 2016

@author: admin
"""
from infodens.feature_extractor.feature_extractor import featid, Feature_extractor
from scipy import sparse


class Surface_features(Feature_extractor):
    
    @featid(1)    
    def averageWordLength(self, argString, preprocessReq=0):
        '''Find average word length of every sentence and return list. '''

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            self.preprocessor.getSentCount()
            self.preprocessor.gettokenizeSents()
            return 1

        aveWordLen = sparse.lil_matrix((self.preprocessor.getSentCount(), 1))

        i = 0
        for sentence in self.preprocessor.gettokenizeSents():
            if len(sentence) is 0:
                aveWordLen[i] = 0
            else:
                length = sum([len(s) for s in sentence])
                aveWordLen[i] = (float(length) / len(sentence))
            i += 1

        return aveWordLen, "Average sentence length"

    @featid(10)
    def sentenceLength(self, argString, preprocessReq=0):
        '''Find length of every sentence and return list. '''

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            self.preprocessor.getSentCount()
            self.preprocessor.gettokenizeSents()
            return 1

        sentLen = sparse.lil_matrix((self.preprocessor.getSentCount(),1))
        i = 0
        for sentence in self.preprocessor.gettokenizeSents():
            sentLen[i] = (len(sentence))
            i += 1

        return sentLen

    @featid(8)
    def parseTreeDepth(self, argString, preprocessReq=0):
        '''Find depth of every sentence's parse tree and return list. '''
        if preprocessReq:
            # Request all preprocessing functions to be prepared
            self.preprocessor.getParseTrees()
            return 1

    @featid(2)
    def syllableRatio(self, argString, preprocessReq=0):
        '''
        We approximate this feature by counting the number of vowel-sequences
        that are delimited by consonants or space in a word, normalized by the number of tokens
        in the chunk
        '''
        if preprocessReq:
            # Request all preprocessing functions to be prepared
            self.preprocessor.getSentCount()
            self.preprocessor.gettokenizeSents()
            return 1

        vowels = ['a', 'e', 'i', 'o', 'u']
        sylRatios = sparse.lil_matrix((self.preprocessor.getSentCount(), 1))
        j = 0
        for sentence in self.preprocessor.gettokenizeSents():
            sylCount = 0
            for word in sentence:
                word2List = list(word)
                for i in range(len(word2List)-1):
                    if word2List[i] in vowels and word2List[i+1] not in vowels:
                        sylCount += 1

            if len(sentence) is 0:
                sylRatios[j] = 0
            else:
                sylRatios[j] = (float(sylCount)/len(sentence))
            j += 1

        return sylRatios
