from infodens.feature_extractor import feature_manager as featman
from infodens.preprocessor.preprocess_services import Preprocess_Services
from infodens.classifier import classifier_manager
from infodens.formater import format
from infodens.controller.configurator import Configurator
import os.path


class Controller:
    """Read and parse the config file, init a FeatureManager,
     and init a classifier manager. Handle output. """

    def __init__(self, configFiles=None):
        self.configFiles = configFiles
        self.configurators = []

        # classification parameters are fixed across Multilingual runs
        self.inputClasses = ""
        self.cv_folds = 1
        self.classifiersList = []
        self.persistModelFile = ""
        self.threadsCount = 1
        self.featOutput = ""
        self.featOutFormat = ""
        self.classifReport = ""

        # array format of dataset and labels for classifying
        self.numSentences = 0
        self.extractedFeats = []
        self.featDescriptors = []
        self.classesList = []

    def parseMergeConfigs(self):

        allFeats = []
        for config in self.configurators:
            allFeats.append(config.featureIDs)

            # Policy is to be the greatest
            if config.cv_folds > self.cv_folds:
                self.cv_folds = config.cv_folds
            if config.threadsCount > self.threadsCount:
                self.threadsCount = config.threadsCount

            # Only once or last instance
            # Todo: report conflicts
            if config.inputClasses:
                self.inputClasses = config.inputClasses
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

        self.classifiersList = list(set(self.classifiersList))

        return allFeats

    def loadConfig(self):
        """Read the config file(s), extract the featureIDs and
        their argument strings.
        """
        statusOK = 1

        # Extract featureID and feature Argument string
        for configFile in self.configFiles:
            with open(configFile) as config:
                # Parse the config file
                configurator = Configurator()
                statusOK = configurator.parseConfig(config)
                self.configurators.append(configurator)

                if not configurator.inputFile and statusOK:
                    print("Error, Missing input files.")
                    exit()

        mergedFeats = self.parseMergeConfigs()

        if not self.inputClasses and (self.classifiersList or self.featOutput):
            print("Error, Missing input files.")
            exit()

        return statusOK, mergedFeats, self.classifiersList

    def classesSentsMismatch(self, inputFile):
        if self.inputClasses:
            # Extract the classed IDs from the given classes file and Check for
            # Length equality with the sentences.
            prep_serv = Preprocess_Services()
            if not self.classesList:
                self.classesList = prep_serv.preprocessClassID(self.inputClasses)
            sentLen = len(prep_serv.preprocessBySentence(inputFile))
            classesLen = len(self.classesList)
            self.numSentences = sentLen
            if sentLen != classesLen:
                return True
        return False

    def manageFeatures(self):
        """Init and call a feature manager. """

        for configurator in self.configurators:
            if self.inputClasses and self.classesSentsMismatch(configurator.inputFile):
                print("Classes and Sentences length differ. Quiting. ")
                return 0

        extractedFeats = []
        for configurator in self.configurators:
            manageFeatures = featman.Feature_manager(self.numSentences, configurator)
            validFeats = manageFeatures.checkFeatValidity()
            if validFeats:
                # Continue to call features
                feats, descriptors = manageFeatures.callExtractors()
                extractedFeats.append(feats)
                self.featDescriptors.append(descriptors)
            else:
                # terminate
                print("Requested Feature ID not available.")
                return 0
        self.extractedFeats = featman.mergeFeats(extractedFeats)
        self.scaleFeatures()
        self.outputFeatures()
        print("Feature Extraction Done. ")

        return 1

    def scaleFeatures(self):
        from sklearn import preprocessing as skpreprocess
        scaler = skpreprocess.MaxAbsScaler(copy=False)
        self.extractedFeats = scaler.fit_transform(self.extractedFeats)

    def outputFeatures(self):
        """Output features if requested."""

        if self.featOutput:
            formatter = format.Format(self.extractedFeats, self.classesList,
                                      self.featDescriptors)
            # if format is not set in config, will use a default libsvm output.
            formatter.outFormat(self.featOutput, self.featOutFormat)
        else:
            print("Feature output was not specified.")

    def classifyFeats(self):
        """Instantiate a classifier Manager then run it. """

        print("Starting classification...")

        if self.inputClasses and self.classifiersList:
            # Classify if the parameters needed are specified
            classifying = classifier_manager.Classifier_manager(
                          self.classifiersList, self.extractedFeats, self.classesList,
                          self.threadsCount, self.cv_folds, self.persistModelFile)

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
                print("Requested Classifier not available.")
                return -1
        else:
            print("Classifier parameters not specified.")
        return 1

