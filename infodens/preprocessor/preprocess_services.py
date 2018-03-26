# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:19:12 2016

@author: admin
"""
import io
import nltk
import subprocess
import os
from collections import Counter
from collections import defaultdict


class Preprocess_Services:

    def __init__(self, srilmBinaries="", kenlmBins="", lang="eng"):
        self.srilmBinaries = srilmBinaries
        self.kenlmBins = kenlmBins
        self.operatingLanguage = lang

    def getBinariesPath(self):
        return self.srilmBinaries, self.kenlmBins

    def preprocessBySentence(self, inputFile):
        """Load the input file which was specified at Init of object."""
        lines = []
        with io.open(inputFile, encoding='utf-8') as f:
            lines = f.readlines()
        return lines

    def preprocessByBlock(self, fileName, blockSize):
        pass

    def preprocessClassID(self, inputClasses):
        """ Extract from each line the integer for class ID. Requires init with Classes file."""
        with io.open(inputClasses, encoding='utf-8') as f:
            lines = f.readlines()
        ids = [float(id) for id in lines]
        return ids

    def getSentTokenizer(self):
        # Tokenize on a sentence level
        # params(sent) returns [tokens]
        return nltk.word_tokenize

    def getPOSTagger(self):
        # Tag on list of tokenize sentences
        # params([tokens],lang=) returns (word,POStag)
        return nltk.pos_tag_sents

    def getLemmatizer(self):
        # Lemmatizes a list of tokens
        # params ([tokens]) returns [lemmas]
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return lemmatizer.lemmatize

    def getFileTokens(self, fileOfTokens):
        """Return tokens from file"""
        tokenizer = self.getSentTokenizer()
        return [tokenizer(sent) for sent in self.preprocessBySentence(fileOfTokens)]

    def dumpTokensTofile(self, dumpFile, tokenSents):
        """ Dump tokens into file"""
        if not os.path.isfile(dumpFile):
            outFile = open(dumpFile, 'w')
            for sent in tokenSents:
                outFile.write("%s\n" % " ".join(sent))
            outFile.close()
        return dumpFile

    def tagPOSfromFile(self, filePOS):
        """ Return POS tagged sentences from given File """
        taggedPOSSents = []
        print("POS tagging..")
        posTagger = self.getPOSTagger()
        tagPOSSents = posTagger(self.getFileTokens(filePOS),lang=self.operatingLanguage)
        for i in range(0, len(tagPOSSents)):
            taggedPOSSents.append([wordAndTag[1] for wordAndTag in tagPOSSents[i]])
        print("POS tagging done.")
        return taggedPOSSents

    def buildNgrams(self, n, freq, tokens, indexing=True):
        """Build and return ngrams from given tokens."""
        ngramsDict = defaultdict(int)
        for sent in tokens:
            ngramsList = list(nltk.ngrams(sent, n))
            for anNgram in ngramsList:
                ngramsDict[anNgram] += 1

        return self.ngramMinFreq(ngramsDict, freq, indexing)

    def languageModelBuilder(self, ngram, corpus, langModelFile, kndiscount=True):
        """Build a language model from given corpus."""

        if not self.kenlmBins and not self.srilmBinaries:
            print("No Language modeling tools provided!")
            exit()

        if not self.kenlmBins:
            #No Kenlm use SRILM
            binaryLib = ("\"{0}ngram-count\"".format(self.srilmBinaries))
            discount = ""
            if kndiscount:
                discount = " -kndiscount"
            commandToRun = "{0} -text {1} -lm {2} -unk -order {3}{4}".format(binaryLib, corpus,
                                                                                     langModelFile, ngram, discount)
        else:
            binaryLib = ("\"{0}lmplz\"".format(self.kenlmBins))
            #print(corpus)
            commandToRun = "{0} -o {1} <{2} >{3}".format(binaryLib, ngram, corpus, langModelFile)

        print("Building Language Model..")
        #print(commandToRun)
        subprocess.call(commandToRun, shell=True)
        print("Language Model done.")

        return langModelFile

    def trainWord2Vec(self, vecSize, corpus, threadsCount):
        import gensim
        print("Training Word2Vec model...")

        class SentIterator(object):
            def __init__(self, corpus):
                self.corpus = corpus

            def __iter__(self):
                with io.open(self.corpus, encoding='utf-8') as corpusFile:
                    for line in corpusFile:
                        yield nltk.word_tokenize(line)

        tokenizedCorpus = SentIterator(corpus)

        model = gensim.models.Word2Vec(tokenizedCorpus, size=vecSize, min_count=1, workers=threadsCount)
        print("Word2Vec model done.")

        return model

    def ngramMinFreq(self, anNgram, freq, indexing=True):
        indexOfngram = 0
        finalNgram = {}
        """Return anNgram with entries that have frequency greater or equal freq"""
        if indexing:
            for k in anNgram.keys():
                if anNgram[k] >= freq:
                    finalNgram[k] = indexOfngram
                    indexOfngram += 1
        else:
            for k in anNgram.keys():
                if anNgram[k] >= freq:
                    finalNgram[k] = anNgram[k]

        return finalNgram, len(finalNgram)
