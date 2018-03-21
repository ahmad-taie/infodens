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
        self.predict = False
        self.predictOrTestFile = ""
        self.trainClasses = ""
        self.testClasses = ""
        self.classifiersList = []
        self.classifierArgs = []
        self.persistModelFile = ""
        self.threadsCount = 1
        self.featOutput = ""
        self.featOutFormat = ""
        self.classifReport = ""

        # array format of dataset and labels for classifying
        self.trainFeats = []
        self.testFeats = []
        self.featDescriptors = []
        self.trainClassesList = []
        self.testClassesList = []

    def parseConfigs(self):

        allFeats = []
        for config in self.configurators:
            allFeats.append(config.featureIDs)

            # Policy is to be the greatest
            if config.threadsCount > self.threadsCount:
                self.threadsCount = config.threadsCount

            # Policy is any or last appearance
            # Possible TODO: report conflicts
            if config.predictSentsFile:
                self.predictOrTestFile = config.predictSentsFile
                self.predict = True
            elif config.testSentsFile:
                self.predictOrTestFile = config.testSentsFile
            if config.trainClasses:
                self.trainClasses = config.trainClasses
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

        mergedFeats = self.parseConfigs()

        print("Requested features: ")
        print(mergedFeats)
        if self.classifiersList:
            print("Requested classifiers: ")
            print(self.classifiersList)

    def loadLabels(self):
        prep_serv = Preprocess_Services()
        self.trainClassesList = prep_serv.preprocessClassID(self.trainClasses)
        if self.testClasses:
            self.testClassesList = prep_serv.preprocessClassID(self.testClasses)

    def scaleFeatures(self):
        from sklearn import preprocessing as skpreprocess
        scaler = skpreprocess.MaxAbsScaler(copy=False)
        self.trainFeats = scaler.fit_transform(self.trainFeats)
        self.testFeats = scaler.fit_transform(self.testFeats)

    def outputTrainFeatures(self):
        """Output features if requested."""
        if not self.featOutput:
            print("Feature output was not specified.")
        else:
            print("Outputting train features..")
            formatter = format.Format(self.trainFeats, self.trainClassesList,
                                      self.featDescriptors)
            # if format is not set in config, will use a default libsvm output.
            formatter.outFormat("{0}_trainFeats".format(self.featOutput), self.featOutFormat)

    def classesSentsMismatch(self, trainFile, testFile):
        prep_serv = Preprocess_Services()
        trainSentsCount = len(prep_serv.preprocessBySentence(trainFile))
        if trainSentsCount != len(self.trainClassesList):
            return True

        if self.testClasses:
            # If no test classes, then predict mode
            testSentsCount = len(prep_serv.preprocessBySentence(testFile))
            if testSentsCount != len(self.testClassesList): 
                return True

        return False

    def manageFeatures(self):
        """Init and call a feature manager. """

        for configurator in self.configurators:
            self.loadLabels()
            if self.classesSentsMismatch(configurator.trainFile, configurator.testSentsFile):
                print("Count of Classes and Sentences differ. Exiting.")
                sys.exit()

        # Start feature extraction
        extractedTrainFeats = []
        extractedTestFeats = []
        for configurator in self.configurators:

            prep_servs = Preprocess_Services(srilmBinaries=configurator.srilmBinPath,
                                                  kenlmBins=configurator.kenlmBinPath,
                                                  lang=configurator.language)

            trainPreprocessor = preprocess.Preprocess(configurator.trainFile, configurator.corpusLM,
                                                      configurator.trainClasses, configurator.language,
                                                      configurator.threadsCount, prep_servs)

            # Preprocessor for the test/predict sentences. Classes file not passsed
            # No peaking.
            testPreprocessor = preprocess.Preprocess(self.predictOrTestFile, configurator.corpusLM,
                                                          "", configurator.language,
                                                          configurator.threadsCount, prep_servs)

            manageFeatures = featman.Feature_manager(configurator.featureIDs,
                                                     configurator.featargs,
                                                     configurator.threadsCount,
                                                     trainPreprocessor, testPreprocessor,
                                                     configurator.trainFeatsFile,
                                                     configurator.testFeatsFile)
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
        print("Final train feature matrix dimensions: {0}".format(trainDims))

        print("Final test feature matrix dimensions: {0}".format(testDims))
        self.scaleFeatures()

        print("Feature Extraction Done. ")

        self.outputTrainFeatures()

    def outputTestFeatures(self, classifierName=""):

        if not self.featOutput and not self.predict:
            print("Feature output was not specified.")
            return 0

        # Output just predicted labels if Predict
        if self.predict:
            print("Outputting features from {0}..".format(classifierName))
            outFile = "{0}_predicted_{1}.label".format(self.predictOrTestFile, classifierName)
            formatter = format.Format(self.testFeats, self.testClassesList)
            formatter.outPredictedLabels(outFile, self.testClassesList)
        # Output also features if requested
        if self.featOutput:
            # Output test Feats
            formatter = format.Format(self.testFeats, self.testClassesList)
            formatter.outFormat("{0}_testFeats".format(self.featOutput), self.featOutFormat)

    def classifyFeats(self):
        """Instantiate a classifier Manager then run it. """

        if not (self.predictOrTestFile and self.classifiersList):
            print("Classifier parameters not specified.")
        else:
            print("Starting classification...")
            # Classify if the parameters needed are specified
            classifying = classifier_manager.Classifier_manager(self.predict,
                          self.classifiersList, self.classifierArgs, self.trainFeats,
                          self.trainClassesList, self.testFeats, self.testClassesList,
                          self.persistModelFile, self.threadsCount)

            validClassifiers = classifying.checkParseClassifier()

            if not validClassifiers:
                # terminate
                print("Requested Classifier not available. Exiting")
                sys.exit()
            else:
                # Continue to call classifiers
                reportOfClassif, labels = classifying.callClassifiers()
                print(reportOfClassif)
                print("Classification done.")
                if self.predict:
                    for i in range(0, len(labels)):
                        self.testClassesList = labels[i]
                        self.outputTestFeatures(self.classifiersList[i])
                else:
                    self.outputTestFeatures()

                # Write output if file specified
                if self.classifReport:
                    with open(self.classifReport, 'w') as classifOut:
                        classifOut.write(reportOfClassif)
                return 0
