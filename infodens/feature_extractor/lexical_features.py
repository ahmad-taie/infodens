# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:16:36 2016

@author: admin
"""
from infodens.feature_extractor.feature_extractor import featid, Feature_extractor
from scipy import sparse


class Lexical_features(Feature_extractor):
    
    def computeDensity(self, taggedSentences, jnrv):

        densities = sparse.lil_matrix((len(taggedSentences), 1))
        i = 0
        for sent in taggedSentences:
            if(len(sent) is 0):
                densities[i] = 0
            else:
                jnrvList = [tagPOS for tagPOS in sent if tagPOS in jnrv]
                densities[i] = (float(len(sent) - len(jnrvList)) / len(sent))
            i += 1

        return densities

    @featid(3)        
    def lexicalDensity(self, argString, preprocessReq=0):
        arguments = argString.split(',')
        filePOS = 0
        if(int(arguments[0])):
            # Use file of tagged sents (last argument)
            filePOS = arguments[-1]
            jnrv = arguments[1:-1]
        else:
            jnrv = arguments[1:]

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            self.preprocessor.getPOStagged(filePOS)
            return 1

        '''
        The frequency of tokens that are not nouns, adjectives, adverbs or verbs. 
        This is computed by dividing the number of tokens tagged with POS tags 
        that do not start with J, N, R or V by the number of tokens in the chunk
        '''
        taggedSents = self.preprocessor.getPOStagged(filePOS)

        return self.computeDensity(taggedSents, jnrv)

    @featid(11)
    def lexicalRichness(self, argString, preprocessReq=0):
        '''
        The ratio of unique tokens in the sentence over the sentence length.
        '''

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            self.preprocessor.getSentCount()
            self.preprocessor.gettokenizeSents()
            return 1

        #TODO : Lemmatize tokens?
        sentRichness = sparse.lil_matrix((self.preprocessor.getSentCount(),1))

        i = 0
        for sentence in self.preprocessor.gettokenizeSents():
            if len(sentence) is 0:
                sentRichness[i] = 0
            else:
                sentRichness[i] = (float(len(set(sentence)))/len(sentence))
            i += 1

        return sentRichness

    @featid(12)
    def lexicalToTokens(self, argString, preprocessReq=0):
        '''
        The ratio of lexical words to tokens in the sentence.
        '''
        arguments = argString.split(',')
        filePOS = 0
        if int(arguments[0]):
            # Use file of tagged sents (last argument)
            filePOS = arguments[-1]
            nonLexicalTags = arguments[1:-1]
        else:
            nonLexicalTags = arguments[1:]

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            self.preprocessor.getSentCount()
            self.preprocessor.getPOStagged(filePOS)
            return 1

        lexicalTokensRatio = sparse.lil_matrix((self.preprocessor.getSentCount(), 1))
        i = 0
        for sentence in self.preprocessor.getPOStagged(filePOS):
            lexicalCount = 0
            for tagPOS in sentence:
                if tagPOS not in nonLexicalTags:
                    lexicalCount += 1
            if len(sentence) is 0:
                lexicalTokensRatio[i] = 0
            else:
                lexicalTokensRatio[i] = (float(lexicalCount) / len(sentence))
            i += 1

        return lexicalTokensRatio
