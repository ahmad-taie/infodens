# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 14:12:49 2016

@author: admin
"""
from infodens.feature_extractor.feature_extractor import featid, Feature_extractor
from scipy import sparse
import numpy as np
from collections import Counter
import operator
from nltk import ngrams
import math
import subprocess
import os
import codecs


class Surprisal_features(Feature_extractor):

    def extractValues(self, srilmOutput, sentCount):
        feats = []
        with codecs.open(srilmOutput, encoding='utf-8') as sents:
            for line in sents:
                if "logprob=" in line:
                    line = str(line).strip().split(" ")
                    if len(feats) < sentCount:
                        tmp = 0.0
                        if line[3] != "undefined":
                            log10Prob = -float(line[3])
                            log2prob = log10Prob/math.log(2,10)
                            tmp = log2prob
                        else:
                            tmp = np.float32(0.0)
                        feats.append(tmp)
        return feats

    def perplexity(self, sentence, sentScore):
        """
        Compute perplexity of a sentence.
        @param sentence One full sentence to score.  Do not include <s> or </s>.
        (Reused from kenlm pyx)

        """
        def as_str(data):
            return data.encode('utf8')

        words = len(as_str(sentence).split()) + 1  # For </s>

        return 2.0 ** (-sentScore / words)

    @featid(20)
    def surplangModelFeat(self, argString, preprocessReq=0):
        '''
        Extracts n-gram Language Model preplexity features.
        '''
        ngramOrder = 3
        langModel = 0
        # Binary1/0,ngramOrder,LMFilePath(ifBinary1)
        arguments = argString.split(',')
        if(int(arguments[0])):
            # Use given langModel
            langModel = "\"{0}\"".format(arguments[-1])

        ngramOrder = int(arguments[1])

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            if not langModel:
                langModel = self.preprocessor.buildLanguageModel(ngramOrder)
            self.preprocessor.getInputFileName()
            self.preprocessor.getBinariesPath()
            return 1

        sentsFile = self.preprocessor.getInputFileName()
        srilmBinary, kenlm = self.preprocessor.getBinariesPath()

        if not langModel:
            langModel = self.preprocessor.buildLanguageModel(ngramOrder)

        if srilmBinary and not kenlm:
            pplFile = "tempLang{0}{1}.ppl".format(os.path.basename(sentsFile), ngramOrder)
            command = "\"{0}ngram\" -order {1} -lm {2} -ppl {3} -debug 1 -unk> {4}".format(srilmBinary, ngramOrder,
                                                                                         langModel, sentsFile, pplFile)

            subprocess.call(command, shell=True)
            probab = self.extractValues(pplFile, self.preprocessor.getSentCount())
            os.remove(pplFile)
            return sparse.lil_matrix(probab)
        else:
            try:
                __import__('imp').find_module('kenlm')
                import kenlm
                model = kenlm.Model(langModel)
                probab = []
                for sent in self.preprocessor.getPlainSentences():
                    log10Prob = model.score(sent, bos=True, eos=True)
                    log2prob = log10Prob / math.log(2, 10)
                    probab.append([log2prob, self.perplexity(sent, log2prob)])
                output = sparse.lil_matrix(probab)
                return output
            except ImportError:
                import pynlpl.lm.lm as pineApple
                arpaLM = pineApple.ARPALanguageModel(langModel)
                probab = []
                print("Using Pineapple")
                for sent in self.preprocessor.gettokenizeSents():
                    log10Prob = arpaLM.score(sent)
                    log2prob = log10Prob / math.log(2, 10)
                    probab.append([log2prob])
                output = sparse.lil_matrix(probab)
                return output

    @featid(21)
    def surplangModelPOSFeat(self, argString, preprocessReq=0):
        '''
        Extracts n-gram POS language model preplexity features.
        '''
        ngramOrder = 3
        langModel = ""
        taggedInput = ""
        taggedCorpus = ""
        # TaggedInput1/0,LM0/1,taggedCorpus0/1,ngramOrder(,TaggedPOSfile(ifTaggedInp1),
        # LMFilePath(ifLM1),taggedCorpus(if LM0&TaggedCorpus1))
        arguments = argString.split(',')
        if int(arguments[0]):
            # Use file of tagged sents (last argument)
            taggedInput = "\"{0}\"".format(arguments[4])
        if int(arguments[1]):
            # Next argument
            langModel = "\"{0}\"".format(arguments[4+int(arguments[0])])
        elif int(arguments[2]):
            taggedCorpus = "\"{0}\"".format(arguments[4+int(arguments[0])])

        ngramOrder = int(arguments[3])

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            if not taggedInput:
                taggedInput = self.prep_servs.dumpTokensTofile(
                                            dumpFile="{0}_tagged_Input.txt".format(self.preprocessor.getInputFileName()),
                                                 tokenSents=self.preprocessor.getPOStagged())
            if not langModel:
                if not taggedCorpus:
                    taggedCorpus = self.prep_servs.dumpTokensTofile(
                                        dumpFile="{0}_tagged_Corpus.txt".format(self.preprocessor.getCorpusLMName()),
                                                     tokenSents=self.prep_servs.tagPOSfromFile(
                                                         self.preprocessor.getCorpusLMName()
                                                     ))

                # If tagged corpus is empty, just use
                langModel = self.preprocessor.buildLanguageModel(ngramOrder, taggedCorpus, False)

            return 1

        if not taggedInput:
            taggedInput = "{0}_tagged_Input.txt".format(self.preprocessor.getInputFileName())
        if not langModel:
            if not taggedCorpus:
                taggedCorpus = "{0}_tagged_Corpus.txt".format(self.preprocessor.getCorpusLMName())
            langModel = self.preprocessor.buildLanguageModel(ngramOrder, taggedCorpus, False)

        srilmBinary, kenlm = self.preprocessor.getBinariesPath()

        if srilmBinary and not kenlm:
            pplFile = "tempLang{0}{1}.ppl".format(os.path.basename(taggedInput), ngramOrder)

            command = "\"{0}ngram\" -order {1} -lm {2} -ppl {3} -debug 1 -unk> {4}".format(srilmBinary, ngramOrder,
                                                                                           langModel, taggedInput, pplFile)

            subprocess.call(command, shell=True)
            probab = self.extractValues(pplFile, self.preprocessor.getSentCount())
            os.remove(pplFile)
            return sparse.lil_matrix(probab)
        else:
            try:
                __import__('imp').find_module('kenlm')
                import kenlm
                model = kenlm.Model(langModel)
                probab = []
                for sent in self.preprocessor.getPOStagged():
                    log10Prob = model.score(sent, bos=True, eos=True)
                    log2prob = log10Prob / math.log(2, 10)
                    probab.append([log2prob, self.perplexity(sent, log2prob)])
                output = sparse.lil_matrix(probab)
                return output
            except ImportError:
                import pynlpl.lm.lm as pineApple
                arpaLM = pineApple.ARPALanguageModel(langModel)
                probab = []
                for sent in self.preprocessor.getPOStagged():
                    log10Prob = arpaLM.score(sent)
                    log2prob = log10Prob / math.log(2, 10)
                    probab.append([log2prob])
                output = sparse.lil_matrix(probab)
                return output
