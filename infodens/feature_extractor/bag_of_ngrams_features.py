
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 14:12:49 2016

@author: admin
"""
from .feature_extractor import featid, Feature_extractor
from collections import Counter
from nltk import ngrams
from scipy import sparse


class Bag_of_ngrams_features(Feature_extractor):

    def ngramArgumentCheck(self, argString, type):
        status = 1
        n = 0
        freq = 0
        filePOS = 0

        argStringList = argString.split(',')
        if argStringList[0].isdigit():
            n = int(argStringList[0])
        else:
            print('Error: n should be an integer')
            status = 0
        if len(argStringList) > 1:
            if argStringList[1].isdigit():
                freq = int(argStringList[1])
            else:
                print('Error: frequency should be an integer')
                status = 0
            #POS file
            if type is "POS" and len(argStringList) > 2:
                if int(argStringList[2]):
                    filePOS = argStringList[3]
        else:
            freq = 1
        return status, n, freq, filePOS

    def preprocessReqHandle(self, type, filePOS):
        if type is "plain":
            listOfSentences = self.preprocessor.gettokenizeSents()
        elif type is "POS":
            listOfSentences = self.preprocessor.getPOStagged(filePOS)
        elif type is "lemma":
            listOfSentences = self.preprocessor.getLemmatizedSents()
        elif type is "mixed":
            listOfSentences = self.preprocessor.getMixedSents()
        else:
            #Assume plain
            listOfSentences = self.preprocessor.gettokenizeSents()

        return 1

    def ngramExtraction(self, ngramType, argString, preprocessReq):
        status, n, freq, filePOS = self.ngramArgumentCheck(argString, ngramType)
        if not status:
            # Error in argument.
            return

        # Handle preprocessing requests and exit
        if preprocessReq:
            self.preprocessReqHandle(type, filePOS)
            return 1

        if ngramType is "plain":
            listOfSentences = self.preprocessor.gettokenizeSents()
        elif ngramType is "POS":
            listOfSentences = self.preprocessor.getPOStagged(filePOS)
        elif ngramType is "lemma":
            listOfSentences = self.preprocessor.getLemmatizedSents()
        elif ngramType is "mixed":
            listOfSentences = self.preprocessor.getMixedSents()
        else:
            #Assume plain
            listOfSentences = self.preprocessor.gettokenizeSents()

        finNgram, numberOfFeatures = self.preprocessor.prep_servs.buildNgrams(n, freq, listOfSentences)

        print("Ngrams built.")

        if numberOfFeatures == 0:
            print("Cut-off too high, no ngrams passed it.")
            return []

        ngramFeatures = sparse.lil_matrix((len(listOfSentences), numberOfFeatures))

        print("Extracting ngram feats.")

        for i in range(len(listOfSentences)):
            ngramsVocab = Counter(ngrams(listOfSentences[i], n))
            lenSent = len(ngramsVocab)

            for ngramEntry in ngramsVocab:
                ## Keys
                ngramIndex = finNgram.get(ngramEntry, -1)
                if ngramIndex >= 0:
                    ngramFeatures[i, ngramIndex] = round((float(ngramsVocab[ngramEntry]) / lenSent), 2)

        print("Finished ngram features.")
        ngramLength = "Ngram feature vector length: " + str(numberOfFeatures)
        print(ngramLength)

        return ngramFeatures, "{0} ngrams with arguments: {1} . Features: {2}".format(
            ngramType, argString, finNgram)

    @featid(4)
    def ngramBagOfWords(self, argString, preprocessReq=0):
        '''
        Extracts n-gram bag of words features.
        '''
        return self.ngramExtraction("plain", argString, preprocessReq)

    @featid(5)
    def ngramBagOfPOS(self, argString, preprocessReq=0):
        '''
        Extracts n-gram POS bag of words features.
        '''
        return self.ngramExtraction("POS", argString, preprocessReq)

    @featid(6)
    def ngramBagOfMixedWords(self, argString, preprocessReq=0):
        '''
        Extracts n-gram mixed bag of words features.
        '''
        return self.ngramExtraction("mixed", argString, preprocessReq)

    @featid(7)
    def ngramBagOfLemmas(self, argString, preprocessReq=0):
        '''
        Extracts n-gram lemmatized bag of words features.
        '''
        return self.ngramExtraction("lemma", argString, preprocessReq)


