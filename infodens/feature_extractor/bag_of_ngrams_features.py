
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 14:12:49 2016

@author: admin
"""
from .feature_extractor import featid, Feature_extractor
from collections import Counter
from nltk import ngrams
from scipy import sparse
import argparse
import sys


class Bag_of_ngrams_features(Feature_extractor):

    def ngramArgumentCheck(self, args, ngramType):

        proc_train = ""
        proc_test = ""
        parser = argparse.ArgumentParser(description='Bag of ngrams args')
        parser.add_argument("-train", help="Path for file to build ngram vector from.",
                            type=str, default="")
        if ngramType != "plain":
            parser.add_argument("-proc_train", help="Path for POS/Lemma tagged train sentences.",
                                type=str, default="")
            parser.add_argument("-proc_test", help="Path for POS/lemma tagged test sentences.",
                                type=str, default="")
        parser.add_argument("-ngram", help="Order of ngram.",
                            type=int, default=1)
        parser.add_argument("-cutoff", help="Min. Cutoff for ngram.",
                            type=int, default=1)

        argsOut = parser.parse_args(args.split())
        if ngramType != "plain":
            proc_train = argsOut.proc_train
            proc_test = argsOut.proc_test

        return argsOut.ngram, argsOut.cutoff, argsOut.train, proc_train, proc_test

    def preprocessReqHandle(self, typeNgram, taggedInp, taggedTest):
        if typeNgram is "plain":
            self.preprocessor.gettokenizeSents()
            self.testPreprocessor.gettokenizeSents()
        elif typeNgram is "POS":
            if not taggedInp:
                self.preprocessor.getPOStagged()
            if not taggedTest:
                self.testPreprocessor.getPOStagged()
        elif typeNgram is "lemma":
            if not taggedInp:
                self.preprocessor.getLemmatizedSents()
            if not taggedTest:
                self.testPreprocessor.getLemmatizedSents()
        elif typeNgram is "mixed":
            if not taggedInp:
                self.preprocessor.getMixedSents()
            if not taggedTest:
                self.testPreprocessor.getMixedSents()
        else:
            #Assume plain
            self.preprocessor.gettokenizeSents()
            self.testPreprocessor.gettokenizeSents()

        return 1

    def extractNgram(self, listOfSentences, n, numberOfFeatures, finNgram):

        ngramFeatures = sparse.lil_matrix((len(listOfSentences), numberOfFeatures))
        for i in range(len(listOfSentences)):
            ngramsVocab = Counter(ngrams(listOfSentences[i], n))
            lenSent = len(ngramsVocab)

            for ngramEntry in ngramsVocab:
                ## Keys
                ngramIndex = finNgram.get(ngramEntry, -1)
                if ngramIndex >= 0:
                    ngramFeatures[i, ngramIndex] = round((float(ngramsVocab[ngramEntry]) / lenSent), 2)

        return ngramFeatures

    def ngramExtraction(self, ngramType, argString, preprocessReq):
        n, freq, trainTokens, taggedInp, taggedTest = self.ngramArgumentCheck(argString, ngramType)

        # Handle preprocessing requests and exit
        if preprocessReq:
            return self.preprocessReqHandle(ngramType, taggedInp, taggedTest)

        # Sentences to get ngrams for
        if ngramType is "plain":
            listOfSentences = self.preprocessor.gettokenizeSents()
            testListOfSentences = self.testPreprocessor.gettokenizeSents()
        elif ngramType is "POS":
            listOfSentences = self.preprocessor.getPOStagged(taggedInp)
            testListOfSentences = self.testPreprocessor.getPOStagged(taggedInp)
        elif ngramType is "lemma":
            listOfSentences = self.preprocessor.getLemmatizedSents(taggedInp)
            testListOfSentences = self.testPreprocessor.getLemmatizedSents(taggedInp)
        elif ngramType is "mixed":
            listOfSentences = self.preprocessor.getMixedSents(taggedInp)
            testListOfSentences = self.testPreprocessor.getMixedSents(taggedInp)
        else:
            #Assume plain
            listOfSentences = self.preprocessor.gettokenizeSents()
            testListOfSentences = self.testPreprocessor.gettokenizeSents()

        if not trainTokens:
            trainSentences = listOfSentences
        else:
            # Given file with tokens, extract tokens
            trainSentences = self.preprocessor.prep_servs.getFileTokens(trainTokens)

        finNgram, numberOfFeatures = self.preprocessor.\
                                    prep_servs.buildNgrams(n, freq, trainSentences)

        print("Ngrams built.")

        if numberOfFeatures == 0:
            print("Cut-off too high, no ngrams passed it.")
            sys.exit()

        print("Extracting ngram feats.")
        trainFeatures = self.extractNgram(listOfSentences, n, numberOfFeatures, finNgram)
        testFeatures = self.extractNgram(testListOfSentences, n, numberOfFeatures, finNgram)

        print("Finished ngram features.")
        ngramLength = "Ngram feature vector length: " + str(numberOfFeatures)
        print(ngramLength)

        ngramDescrip = "\r\n".join(["{0}: {1}".format(gram[1], gram[0])
                                    for gram in finNgram.items()])

        return trainFeatures, testFeatures, "{0} ngrams with arguments: {1}:\r\n{2}".format(
            ngramType, argString, ngramDescrip)

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


