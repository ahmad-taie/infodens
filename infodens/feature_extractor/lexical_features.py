# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:16:36 2016

@author: admin
"""
from infodens.feature_extractor.feature_extractor import featid, Feature_extractor
from scipy import sparse
import argparse


class Lexical_features(Feature_extractor):
    
    def computeDensity(self, taggedSentences, jnrv):

        densities = []
        for sent in taggedSentences:
            if len(sent) is 0:
                densities.append(0)
            else:
                jnrvList = [tagPOS for tagPOS in sent if tagPOS in jnrv]
                densities.append(float(len(sent) - len(jnrvList)) / len(sent))

        return sparse.lil_matrix(densities).transpose()

    def parsePOSArgs(self, args):

        parser = argparse.ArgumentParser(description='Lexical density args')
        parser.add_argument("-pos_train", help="Path for POS tagged train sentences.",
                            type=str, default="")
        parser.add_argument("-pos_test", help="Path for POS tagged test sentences.",
                            type=str, default="")
        parser.add_argument("-pos_tags", help="Comma separated list of POS tags.",
                            type=str, default="")

        argsOut = parser.parse_args(args.split())
        return argsOut.pos_train, argsOut.pos_test, argsOut.pos_tags.split(",")

    @featid(3)        
    def lexicalDensity(self, argString, preprocessReq=0):
        trainPOS, testPOS, jnrv = self.parsePOSArgs(argString)

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            self.preprocessor.getPOStagged(trainPOS)
            self.testPreprocessor.getPOStagged(testPOS)
            return 1

        '''
        The frequency of tokens that are not nouns, adjectives, adverbs or verbs. 
        This is computed by dividing the number of tokens tagged with POS tags 
        that do not start with J, N, R or V by the number of tokens in the chunk
        '''
        trainLexDens = self.computeDensity(self.preprocessor.getPOStagged(trainPOS), jnrv)
        testLexDens = self.computeDensity(self.testPreprocessor.getPOStagged(testPOS), jnrv)

        return trainLexDens, testLexDens, "Lexical density"

    def getLexicalRichness(self, sents):
        sentRichness = []
        for sentence in sents:
            if len(sentence) is 0:
                sentRichness.append(0)
            else:
                sentRichness.append(float(len(set(sentence)))/len(sentence))
        return sparse.lil_matrix(sentRichness).transpose()

    @featid(11)
    def lexicalRichness(self, argString, preprocessReq=0):
        '''
        The ratio of unique tokens in the sentence over the sentence length.
        '''

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            self.testPreprocessor.gettokenizeSents()
            self.preprocessor.gettokenizeSents()
            return 1

        #TODO : Lemmatize tokens?
        trainSentRichness = self.getLexicalRichness(self.preprocessor.gettokenizeSents())
        testSentRichness = self.getLexicalRichness(self.testPreprocessor.gettokenizeSents())

        return trainSentRichness, testSentRichness, "Type token ratio"

    def getLexicalToTokens(self, sents, lexicalTags):

        lexicalTokensRatio = []
        for sentence in sents:
            lexicalCount = 0
            for tagPOS in sentence:
                if tagPOS in lexicalTags:
                    lexicalCount += 1
            if len(sentence) is 0:
                lexicalTokensRatio.append(0)
            else:
                lexicalTokensRatio.append(float(lexicalCount) / len(sentence))

        return sparse.lil_matrix(lexicalTokensRatio).transpose()

    @featid(12)
    def lexicalToTokens(self, argString, preprocessReq=0):
        '''
        The ratio of lexical words to tokens in the sentence.
        '''
        trainPOS, testPOS, lexicalTags = self.parsePOSArgs(argString)

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            self.preprocessor.getPOStagged(trainPOS)
            self.testPreprocessor.getPOStagged(testPOS)
            return 1

        trainLexTokRatio = self.getLexicalToTokens(self.preprocessor.getPOStagged(trainPOS), lexicalTags)
        testLexTokRatio = self.getLexicalToTokens(self.testPreprocessor.getPOStagged(testPOS), lexicalTags)

        return trainLexTokRatio, testLexTokRatio, "Ratio of lexical words to tokens"
