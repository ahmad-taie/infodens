from infodens.feature_extractor import feature_manager as featman
from infodens.preprocessor import preprocess
from infodens.preprocessor.preprocess_services import Preprocess_Services
from infodens.classifier import classifier_manager
from infodens.formater import format
from infodens.controller.configurator import Configurator
import os.path
import sys


class Controller:
    """Read and parse the config file, init a FeatureManager,
     and init a classifier manager. Handle output. """

    def __init__(self, configFiles=None):
        self.configFiles = configFiles
        self.configurators = []

        # classification parameters are fixed across Multilingual runs
        self.inputClasses = ""
        self.testClasses = ""
        self.cv_folds = 1
        self.cv_Percent = 0
        self.classifiersList = []
        self.classifierArgs = []
        self.persistModelFile = ""
        self.threadsCount = 1
        self.featOutput = ""
        self.featOutFormat = ""
        self.classifReport = ""

        # array format of dataset and labels for classifying
        self.trainSentsCount = 0
        self.testSentsCount = 0
        self.trainFeats = []
        self.testFeats = []
        self.featDescriptors = []
        self.trainClassesList = []
        self.testClassesList = []

    def mergeConfigs(self):

        allFeats = []
        for config in self.configurators:
            allFeats.append(config.featureIDs)

            # Policy is to be the greatest
            if config.cv_folds > self.cv_folds:
                self.cv_folds = config.cv_folds
            if config.cv_Percent > self.cv_Percent:
                self.cv_Percent = config.cv_Percent
            if config.threadsCount > self.threadsCount:
                self.threadsCount = config.threadsCount
            if config.persistOnFull:
                self.persistOnFull = True

            # Policy is any or last appearance
            # Possible TODO: report conflicts
            if config.inputClasses:
                self.inputClasses = config.inputClasses
            if config.testClasses:
                self.testClasses = config.testClasses
            if config.featOutput:
                self.featOutput = config.featOutput
            if config.featOutFormat:
                self.featOutFormat = config.featOutFormat
            if config.classifReport:
                self.classifReport = config.classifReport
            if config.persistClassif:
                self.persistModelFile = config.persistClassif

            # Classifiers in different configs are merged
            if config.classifiersList:
                self.classifiersList.extend(config.classifiersList)
                self.classifierArgs.extend(config.classifierArgs)

        return allFeats

    def loadConfig(self):
        """Read the config file(s), extract the featureIDs and
        their argument strings.
        """
        # Extract featureID and feature Argument string
        for configFile in self.configFiles:
            # Parse the config file
            configurator = Configurator(configFile)
            configurator.parseConfig()
            self.configurators.append(configurator)

        mergedFeats = self.mergeConfigs()

        # No sents classes provided but output or classification requested
        if not self.inputClasses and (self.classifiersList or self.featOutput):
            print("Error, Missing input classes file.")
            sys.exit()

        print("Requested features: ")
        print(mergedFeats)
        if self.classifiersList:
            print("Requested classifiers: ")
            print(self.classifiersList)

    def loadClasses(self, inputFile, testFile):
        prep_serv = Preprocess_Services()
        self.trainClassesList = prep_serv.preprocessClassID(self.inputClasses)
        if self.testClasses:
            # If no test classes, then predict mode
            self.testClassesList = prep_serv.preprocessClassID(self.testClasses)
            self.testSentsCount = len(prep_serv.preprocessBySentence(testFile))

        self.trainSentsCount = len(prep_serv.preprocessBySentence(inputFile))

    def classesSentsMismatch(self):
        if (self.trainSentsCount != len(self.trainClassesList)) or\
                (self.testSentsCount != len(self.testClassesList)):
            return True
        else:
            return False

    def manageFeatures(self, returnFeats=False):
        """Init and call a feature manager. """

        for configurator in self.configurators:
            self.loadClasses(configurator.inputFile, configurator.testSentsFile)
            if self.classesSentsMismatch():
                print("Count of Classes and Sentences differ. Exiting.")
                sys.exit()

        extractedTrainFeats = []
        extractedTestFeats = []
        for configurator in self.configurators:

            prep_servs = Preprocess_Services(srilmBinaries=configurator.srilmBinPath,
                                                  kenlmBins=configurator.kenlmBinPath,
                                                  lang=configurator.language)

            trainPreprocessor = preprocess.Preprocess(configurator.inputFile, configurator.corpusLM,
                                                      configurator.inputClasses, configurator.language,
                                                      configurator.threadsCount, prep_servs)
            # Preprocessor for the test sentences
            testPreprocessor = preprocess.Preprocess(configurator.testSentsFile, configurator.corpusLM,
                                                          configurator.testClasses, configurator.language,
                                                          configurator.threadsCount, prep_servs)

            manageFeatures = featman.Feature_manager(configurator.featureIDs,
                                                     configurator.featargs,
                                                     configurator.threadsCount,
                                                     trainPreprocessor, testPreprocessor)
            validFeats = manageFeatures.checkFeatValidity()
            if validFeats:
                # Continue to call features
                trainFeats, testFeats, descriptors = manageFeatures.callExtractors()
                extractedTrainFeats.append(trainFeats)
                extractedTestFeats.append(testFeats)
                self.featDescriptors.append(descriptors)
            else:
                # terminate
                print("Requested Feature ID not available. Exiting.")
                sys.exit()

        self.trainFeats, trainDims = featman.mergeFeats(extractedTrainFeats)
        self.testFeats, testDims = featman.mergeFeats(extractedTestFeats)
        print("Final train feature matrix dimensions: {0}\r\n"
              "Final test feature matrix dimensions: {1}".format(trainDims, testDims))
        self.scaleFeatures()

        print("Feature Extraction Done. ")

        if returnFeats:
            # TODO: Done for the infopredict, will be removed
            return self.trainFeats
        self.outputFeatures()

    def scaleFeatures(self):
        from sklearn import preprocessing as skpreprocess
        scaler = skpreprocess.MaxAbsScaler(copy=False)
        self.trainFeats = scaler.fit_transform(self.trainFeats)
        self.testFeats = scaler.fit_transform(self.testFeats)

    def outputFeatures(self):
        """Output features if requested."""

        if self.featOutput:
            formatter = format.Format(self.trainFeats, self.trainClassesList,
                                      self.featDescriptors)
            # if format is not set in config, will use a default libsvm output.
            formatter.outFormat("testFeats_{0}".format(self.featOutput), self.featOutFormat)

            # Output test Feats
            formatter = format.Format(self.testFeats, self.testClassesList,
                                      self.featDescriptors)
            formatter.outFormat("testFeats_{0}".format(self.featOutput), self.featOutFormat)
        else:
            print("Feature output was not specified.")

    def classifyFeats(self):
        """Instantiate a classifier Manager then run it. """

        if self.testClasses and self.classifiersList:
            print("Starting classification...")
            # Classify if the parameters needed are specified
            classifying = classifier_manager.Classifier_manager(
                          self.classifiersList, self.classifierArgs, self.trainFeats,
                          self.trainClassesList, self.testFeats, self.testClassesList,
                          self.persistModelFile, self.threadsCount)

            validClassifiers = classifying.checkParseClassifier()

            if validClassifiers:
                # Continue to call classifiers
                reportOfClassif = classifying.callClassifiers()
                print(reportOfClassif)
                print("Classification done.")
                # Write output if file specified
                if self.classifReport:
                    with open(self.classifReport, 'w') as classifOut:
                        classifOut.write(reportOfClassif)
                return 0
            else:
                # terminate
                print("Requested Classifier not available. Exiting")
                sys.exit()
        else:
            print("Classifier parameters not specified.")
