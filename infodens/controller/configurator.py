import os.path
import sys
import configparser
from collections import OrderedDict


# Use custom dict for multi-valued entries
class MultiOrderedDict(OrderedDict):
    def __setitem__(self, key, value):
        if isinstance(value, list) and key in self:
            self[key].extend(value)
        else:
            super(MultiOrderedDict, self).__setitem__(key, value)
            # super().__setitem__(key, value) in Python 3


class Configurator:
    """Read and parse the config file and return a config object """

    def __init__(self, configFile=None):
        self.configFile = configFile

        self.inputFile = ""
        self.inputClasses = ""
        self.testSentsFile = ""
        self.testClasses = ""
        self.predictSentsFile = ""

        self.featureIDs = []
        self.featargs = []
        self.classifiersList = []
        self.classifierArgs = []
        self.persistClassif = ""
        self.classifReport = ""
        self.corpusLM = ""
        self.featOutput = ""
        self.featOutFormat = ""
        self.threadsCount = 1
        self.language = 'eng'
        self.srilmBinPath = ""
        self.kenlmBinPath = ""

    def getParams(self, config):

        # Read the Input values
        if "Input" in config:
            self.inputFile = config["Input"].get("input file", "")
            self.inputClasses = config["Input"].get("input classes", "")
            self.testSentsFile = config["Input"].get("test file", "")
            self.predictSentsFile = config["Input"].get("predict file", "")
            self.testClasses = config["Input"].get("test classes", "")
            if not self.inputFile or not self.inputClasses\
                    or not (self.testSentsFile or self.predictSentsFile):
                print("Error: Files missing. Requires: Input file and "
                      "classes and test/predict file.")
                sys.exit()
            if (self.testSentsFile or self.testClasses) and self.predictSentsFile:
                print("Error: Only Test or Predict in one run.")
                sys.exit()
            if not self.predictSentsFile and not (self.testSentsFile and self.testClasses):
                print("Error: Missing test file or classes.")
                print("If predicting use \"predict file:\" argument.")
                sys.exit()

            self.corpusLM = config["Input"].get("training corpus", "")
            self.language = config["Input"].get("language", "en")
        else:
            print("Error: Input section missing.")
            sys.exit()

        # Read any given settings
        if "Settings" in config:
            self.threadsCount = config["Settings"].getint("threads", 1)
            self.kenlmBinPath = config["Settings"].get("kenlm", "")
            self.srilmBinPath = config["Settings"].get("srilm", "")

        # Read the output values
        if "Output" in config:
            self.classifReport = config["Output"].get("classifier report", "")

            self.persistClassif = config["Output"].get("persist models", "")

            outFeats = config["Output"].get("output features", "")
            if outFeats:
                outFeats = outFeats.strip().split()
                if len(outFeats) >= 2:
                    self.featOutput = outFeats[0]
                    self.featOutFormat = outFeats[1:]
                else:
                    print("Config Error: Feature output format not specified.")
                    sys.exit()

        # Load feature IDs and their args
        if "Features" in config:
            for feat in config["Features"]:
                # Handle repeated feats with different args
                args = config["Features"].get(feat, "")
                if args:
                    args = args.split("\n")
                    for arg in args:
                        self.featureIDs.append(int(feat))
                        self.featargs.append(arg)
                else:
                    self.featureIDs.append(int(feat))
                    self.featargs.append("")
        else:
            print("No features requested.")

        if "Classifiers" in config:
            for classif in config["Classifiers"]:
                self.classifiersList.append(classif)
                self.classifierArgs.append(config["Classifiers"].get(classif, ""))

    def parseConfig(self):
        """Parse the config file lines.
              """
        # Init a configParser object
        config = configparser.ConfigParser(dict_type=MultiOrderedDict,
                                           strict=False, allow_no_value=True)
        config.optionxform = lambda option: option

        config.read(self.configFile)
        self.getParams(config)

