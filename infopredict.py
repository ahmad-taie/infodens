import sys
from infodens.controller.controller import Controller
import joblib


def predictLabels(models, config, outFile):
    # Init a Controller.
    control = Controller(config)
    # Load the config file
    status, featIds, classifiersList = control.loadConfig()
    # MAIN PROCESS (Extract all features)
    if status != 0:
        print("Requested features: ")
        print(featIds)
        # Manages feature Extraction
        feats = control.manageFeatures(returnFeats=True)

        for model in models:
            end = model.rfind('_')
            modelName = model[:end]
            print("Predicting with {0}..".format(modelName))
            clf = joblib.load(model)
            labels = clf.predict(feats)
            outName = "{0}_{1}".format(modelName, outFile)
            import numpy
            numpy.savetxt(outName, labels)
            print("Done.")
        print("All Done.")


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

    if len(sys.argv) > 3:
        config.append(sys.argv[1])
        outFile = sys.argv[2]
        models = sys.argv[3:]
        print("Config input : {0}".format(config))
        print("Outfile name {0}".format(outFile))
        print("Models to use : {0}".format(models))
        predictLabels(models, config, outFile)
    else:
        print("Too few parameters. Exiting.")
        exit()

