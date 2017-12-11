import sys
from infodens.preprocessor.preprocess_services import Preprocess_Services
from infodens.classifier.fast_text import Fast_text


def loader(inpSents, inpLabels):

    fileLoader = Preprocess_Services()

    sents = fileLoader.preprocessBySentence(inpSents)
    labls = fileLoader.preprocessClassID(inpLabels)

    return sents, labls


if __name__ == '__main__':
    import platform
    print("Running Python {0}".format(platform.python_version()))

    # Params:
    # 1- Config. Contains file to predict and features
    config = []
    outFile = ""
    models = []
    # 2- outfile. Name of output file for labels
    # 3- model(s_. The models to unpickle and use for labeling

    if len(sys.argv) > 4:
        inputFile = sys.argv[1]
        inputClasses = sys.argv[2]
        cv_percent = float(sys.argv[3])
        folds = int(sys.argv[4])

        print("Input sents : {0}".format(inputFile))
        print("Outfile name {0}".format(inputClasses))
        print("CV percentage: {0} for {1} folds".format(cv_percent, folds))
    else:

    sents, labels = loader(inputFile, inputClasses)

    classif = Fast_text(sents, labels, threads=1, nCrossValid=folds,
                       cv_Percent=cv_percent)
    report = classif.runClassifier()
    print(report)

