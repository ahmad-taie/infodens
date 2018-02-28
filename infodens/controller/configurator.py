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
        self.featureIDs = []
        self.featargs = []
        self.inputClasses = []
        self.classifiersList = []
        self.classifierArgs = []
        self.persistClassif = ""
        self.persistOnFull = False
        self.inputFile = ""
        self.classifReport = ""
        self.corpusLM = ""
        self.featOutput = ""
        self.featOutFormat = ""
        self.threadsCount = 1
        self.language = 'eng'
        self.srilmBinPath = ""
        self.kenlmBinPath = ""
        self.cv_folds = 1
        self.cv_Percent = 0

    def getParams(self, config):

        # Read the Input values
        if "Input" in config:
            self.inputFile = config["Input"].get("input file", "")
            if not self.inputFile:
                print("Error: Missing input files.")
                sys.exit()

            self.inputClasses = config["Input"].get("input classes", "")
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

            configLine = config["Settings"].get("folds", "1 0")
            configLine = configLine.strip().split()
            if configLine[0].isdigit():
                folds = int(configLine[0])
                if folds > 0:
                    self.cv_folds = folds
                    if len(configLine) > 1:
                        self.cv_Percent = float(configLine[1])
                else:
                    print("Number of folds is not a positive integer.")
                    sys.exit()
            else:
                print("Number of folds is not a positive integer.")
                sys.exit()

        # Read the output values
        if "Output" in config:
            self.classifReport = config["Output"].get("classifier report", "")

            configLine = config["Output"].get("persist models", "")
            configLine = configLine.strip().split()
            if configLine:
                self.persistClassif = configLine[0]
                if len(configLine) > 1 and "f" in configLine[1]:
                    self.persistOnFull = True
                    print("Model will be persisted on full input. ")

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
                        #print(arg)
                        self.featargs.append(arg)
                else:
                    self.featureIDs.append(int(feat))
                    self.featargs.append(args)

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

