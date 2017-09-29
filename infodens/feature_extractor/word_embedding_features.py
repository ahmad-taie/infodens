from infodens.feature_extractor.feature_extractor import featid, Feature_extractor
import numpy as np
from scipy import sparse
import gensim


class Word_embedding_features(Feature_extractor):
    
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
            return 1

        # Uses language Model from config File
        if not modelFile:
            model = self.preprocessor.getWord2vecModel(vecSize)
        else:
            binary = False
            if modelFile.endswith(".bin"):
                binary = True
            model = gensim.models.KeyedVectors.load_word2vec_format(modelFile, binary=binary)

        vecAverages = []

        for sentence in self.preprocessor.gettokenizeSents():
            if len(sentence) is 0:
                #Empty sentence, default is zero vector
                vecAverages.append([0]*vecSize)
            else:
                sentVec = []
                for word in sentence:
                    try:
                        wordVector = model[word]
                        sentVec.append(wordVector)
                    except KeyError:
                        #wordVector = [0]*vecSize
                        # Skip OOV
                        continue
                vecAverages.append(np.mean(sentVec, axis=0))

        return sparse.lil_matrix(vecAverages)


    @featid(34)
    def word2vecMoments(self, argString, preprocessReq=0):
        '''Find average word2vec vector of every sentence. '''

        if len(argString) > 0:
            vecSize = int(argString)
        else:
            #default
            vecSize = 100

        if preprocessReq:
            # Request all preprocessing functions to be prepared
            self.preprocessor.getWord2vecModel(vecSize)
            self.preprocessor.gettokenizeSents()
            return 1

        # Uses language Model from config File
        model = self.preprocessor.getWord2vecModel(vecSize)
        huMoments = []

        for sentence in self.preprocessor.gettokenizeSents():
            vecImage = []
            for word in sentence:
                vecImage.append(model[word])

            import cv2
            huMoments.append(cv2.HuMoments(cv2.moments(np.asarray(vecImage))).flatten())

        return sparse.lil_matrix(huMoments)
