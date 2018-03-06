import importlib
import os
import sys, inspect
from joblib import Parallel, delayed
import itertools
from scipy import sparse


def runFeatureMethod(mtdCls, featureID, preprocessor, testPreprocessor,
                     featureName, featureArgs, preprocessReq=0):
    """ Run the given feature extractor. """
    instance = mtdCls(preprocessor, testPreprocessor)
    methd = getattr(instance, featureName)
    feat = methd(featureArgs, preprocessReq)
    if not preprocessReq:
        feateX = "Extracted feature: {0} - {1}".format(featureID, featureName)
        print(feateX)
        # Not a tuple then add a feature descriptor
        if not isinstance(feat, tuple):
            return feat, "FeatID {0} - {1} with arguments: {2}".format(featureID, featureName,
                                                                       featureArgs)
    return feat


def mergeFeats(featMatrices):

    output = sparse.hstack(featMatrices, "lil")

    return output, output.get_shape()


class Feature_manager:
    """ Validate the config feature requests,
    And call the necessary feature extractors.
    """

    def __init__(self, featureIDs, featargs, threadsCount, trainPrep, testPrep):
        self.featureIDs = featureIDs
        self.featureArgs = featargs
        self.preprocessor = trainPrep
        # Preprocessor for the test sentences
        self.testPreprocessor = testPrep
        self.threads = threadsCount
        self.featDescriptors = []

        sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
        import infodens.feature_extractor.feature_extractor as feat_extr
        self.pathname = os.path.dirname(os.path.abspath(feat_extr.__file__))

        # Make class variable
        self.idClassmethod, self.allFeatureIds = self.idClassDictionary()

    def separateFeatAndDescrip(self, featAndDesc):

        trainFeats = []
        testFeats = []
        featIndex = 0

        for tuple in featAndDesc:
            trainFeats.append(tuple[0])
            testFeats.append(tuple[1])

            if tuple[0].get_shape()[1] > 1:
                self.featDescriptors.append("Features {0} to {1}: {2}".format(
                    featIndex,
                    featIndex+tuple[0].get_shape()[1] - 1,
                    tuple[2]))
            else:
                self.featDescriptors.append("Feature {0}: {1}".format(
                    featIndex, tuple[2]))
            featIndex = featIndex + tuple[0].get_shape()[1]

        #print(self.featDescriptors)
        return trainFeats, testFeats

    def checkFeatValidity(self):
        ''' Check if requested feature exists. '''
        for featID in self.featureIDs:
            if featID not in self.allFeatureIds:
                print("The requested feature {0} is not available!".format(featID))
                return 0
        return 1

    def methodsWithDecorator(self, cls):
        '''
        find all methods in clst
        
        '''
        theMethods = {}

        things = inspect.getmembers(cls(None), predicate=inspect.ismethod)

        for method in things:
            if method[0] is not "__init__":
                featFunc = getattr(cls, method[0])
                if "_featid_" in featFunc.__name__:
                    idstart = featFunc.__name__.index("_featid_") + len("_featid_")
                    featid = int(featFunc.__name__[idstart:])
                    theMethods[featid] = method[0]

        return theMethods

    def idClassDictionary(self):
        '''
        for every id chosen, find the class that has the method and pair them in a dictionary.
        '''
        possFeatureClasses = set([os.path.splitext(module)[0]
                                  for module in os.listdir(self.pathname) if module.endswith('.py')])
        possFeatureClasses.discard('feature_extractor')
        possFeatureClasses.discard('__init__')
        possFeatureClasses.discard('feature_manager')

        # All feature Ids
        allFeatureIds = {};  featureIds = {};  idClassmethod = {}

        for eachName in possFeatureClasses:
            
            modd = __import__('feature_extractor.'+eachName)
            modul = getattr(modd, eachName)
            clsmembers = inspect.getmembers(modul, inspect.isclass)

            if len(clsmembers) > 0:
                clsmembers = [m for m in clsmembers if m[1].__module__.startswith('feature_extractor') and
                             m[0] is not 'Feature_Extractor']
                for i in range(0, len(clsmembers)):
                    featureIds = self.methodsWithDecorator(clsmembers[i][1])
                    allFeatureIds.update(featureIds)
                    idClassmethod.update({k: clsmembers[i][1] for k in featureIds.keys()})

        return idClassmethod, allFeatureIds

    def callExtractors(self):
        '''Extract all feature Ids and names.  '''

        # Gather preprocessor requests first
        for i in range(len(self.featureIDs)):
            runFeatureMethod(self.idClassmethod[self.featureIDs[i]],
                             self.featureIDs[i], self.preprocessor,
                             self.testPreprocessor,
                             self.allFeatureIds[self.featureIDs[i]],
                             self.featureArgs[i], preprocessReq=1)

        # Use the minimum of threads and number of requested features
        # Don't allocate unneeded processes
        threadsToUse = len(self.featureIDs) if len(self.featureIDs) < self.threads else self.threads
        featuresExtracted = Parallel(n_jobs=threadsToUse, mmap_mode='r')(delayed(runFeatureMethod)(
                                                        self.idClassmethod[self.featureIDs[i]],
                                                        self.featureIDs[i],
                                                        self.preprocessor,
                                                        self.testPreprocessor,
                                                        self.allFeatureIds[self.featureIDs[i]],
                                                        self.featureArgs[i])
                                                       for i in range(len(self.featureIDs)))

        print("All features extracted. ")

        trainFeatures, testFeatures = self.separateFeatAndDescrip(featuresExtracted)

        #Format into scikit format (Each row is a sen)
        trainFeatures = sparse.hstack(trainFeatures, "lil")
        testFeatures = sparse.hstack(testFeatures, "lil")

        return trainFeatures, testFeatures, self.featDescriptors
