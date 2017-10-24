import os.path


class Configurator:
    """Read and parse the config file and return a config object """

    def __init__(self):
        self.featureIDs = []
        self.featargs = []
        self.inputClasses = []
        self.classifiersList = []
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

    def parseOutputLine(self, line):
        status = 1
        startInp = line.index(':')
        outputLine = line[startInp + 1:]
        outputLine = outputLine.strip().split()
        if "classif" in line and not self.classifReport:
            self.classifReport = outputLine[0]
        elif "feat" in line and not self.featOutput:
            if len(outputLine) == 2:
                self.featOutput = outputLine[0]
                self.featOutFormat = outputLine[1]
            elif len(outputLine) == 1:
                self.featOutput = outputLine[0]
            else:
                status = 0
                print("Incorrect number of output params, should be exactly 2")
        else:
            print("Unsupported output type")
            status = 0

        return status

    def parseConfig(self, configFile):
        """Parse the config file lines.      """
        statusOK = 1

        for configLine in configFile:
            configLine = configLine.strip()
            if not statusOK:
                break
            if len(configLine) < 1:
                # Line is empty
                continue
            elif configLine[0] is '#':
                # Line is comment
                continue
            elif "input file" in configLine:
                # Extract input file
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                self.inputFile = configLine[0]
                print("Input file: ")
                print(self.inputFile)
            elif "input class" in configLine:
                # Extract input classes file
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                self.inputClasses = configLine[0]
                #print("Input classes: ")
                #print(self.inputClasses)
            elif "output" in configLine:
                statusOK = self.parseOutputLine(configLine)
            elif "classif" in configLine:
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split(',')
                self.classifiersList = configLine
            elif "training corpus" in configLine:
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                self.corpusLM = configLine[0]
            elif "SRILM" in configLine or "srilm" in configLine:
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip()
                self.srilmBinPath = configLine
                if not os.path.isdir(self.srilmBinPath):
                    statusOK = 0
                    print("Invalid SRILM binaries path.")
                else:
                    self.srilmBinPath = os.path.join(self.srilmBinPath, '')
            elif "kenlm" in configLine:
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip()
                self.kenlmBinPath = configLine
                if not os.path.isdir(self.kenlmBinPath):
                    statusOK = 0
                    print("Invalid KenLm binaries path.")
                else:
                    self.kenlmBinPath = os.path.join(self.kenlmBinPath, '')
            elif "operating language" in configLine:
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                self.language = configLine
                #print(self.language)
            elif "thread" in configLine:
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                if configLine[0].isdigit():
                    threads = int(configLine[0])
                    if threads > 0:
                        #handle single thread case
                        self.threadsCount = threads if threads < 3 else threads-1
                    else:
                        statusOK = 0
                        print("Number of threads is not a positive integer.")
                    #print(self.threadsCount)
                else:
                    statusOK = 0
                    print("Number of threads is not a positive integer.")
            elif "fold" in configLine:
                startInp = configLine.index(':')
                configLine = configLine[startInp + 1:]
                configLine = configLine.strip().split()
                if configLine[0].isdigit():
                    folds = int(configLine[0])
                    if folds > 0:
                        self.cv_folds = folds

                    else:
                        statusOK = 0
                        print("Number of folds is not a positive integer.")
                else:
                    statusOK = 0
                    print("Number of folds is not a positive integer.")
            else:
                params = str(configLine).split(' ', 1)
                if len(params) == 2 or len(params) == 1:
                    if params[0].isdigit():
                        self.featureIDs.append(int(params[0]))
                        if len(params) == 2:
                            self.featargs.append(params[1])
                        else:
                            self.featargs.append([])
                    else:
                        statusOK = 0
                        print("Feature ID is not a Number")
                else:
                    # Incorrect number/value of params
                    statusOK = 0
                    print("Incorrect number of params, max 2 parameters.")

        return statusOK

