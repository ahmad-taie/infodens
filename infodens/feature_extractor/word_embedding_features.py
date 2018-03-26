from infodens.feature_extractor.feature_extractor import featid, Feature_extractor
import numpy as np
from scipy import sparse
import gensim


class Word_embedding_features(Feature_extractor):

    def getSentEmbed(self, sents, model, vecSize):
        avgEmbed = []
        for sentence in sents:
            if len(sentence) is 0:
                #Empty sentence, default is zero vector
                avgEmbed.append([0]*vecSize)
            else:
                sentVec = []
                for word in sentence:
                    if word in model.wv.vocab:
                        wordVector = model[word]
                        sentVec.append(wordVector)
                    else:
                        #wordVector = [0]*vecSize
                        # Skip OOV
                        continue
                if len(sentVec) > 0:
                    avgEmbed.append(np.mean(sentVec, axis=0))
                else:
                    avgEmbed.append([0] * vecSize)
        return sparse.lil_matrix(avgEmbed)
    
    @featid(33)
    def word2vecAverage(self, argString, preprocessReq=0):
        '''Find average word2vec vector of every sentence. '''

        modelFile = ""
        vecSize = 100

        if len(argString) > 0:
            if argString.isdigit():
                vecSize = int(argString)
            else:
                modelFile = argString

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            if not modelFile:
                self.preprocessor.getWord2vecModel(vecSize)
            self.preprocessor.gettokenizeSents()
            self.testPreprocessor.gettokenizeSents()
            return 1

        # Uses language Model from config File
        if not modelFile:
            model = self.preprocessor.getWord2vecModel(vecSize)
        else:
            binary = False
            if modelFile.endswith(".bin"):
                binary = True
            model = gensim.models.KeyedVectors.load_word2vec_format(modelFile, binary=binary)
            vecSize = len(model[next(iter(model.wv.vocab))])

        trainVecAverages = self.getSentEmbed(self.preprocessor.gettokenizeSents(), model, vecSize)
        testVecAverages = self.getSentEmbed(self.testPreprocessor.gettokenizeSents(), model, vecSize)

        return trainVecAverages, testVecAverages, "Average sentence embedding"
