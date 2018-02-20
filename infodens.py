from infodens.controller.controller import Controller
import sys
import os


def infodensRun(configFiles):

    # Init a Controller.
    control = Controller(configFiles)
    # Load the config file
    control.loadConfig()
    # MAIN PROCESS (Extract all features then classify)
    control.manageFeatures()
    control.classifyFeats()


if __name__ == '__main__':
    import platform
    print("Running Python {0}".format(platform.python_version()))

    configs = []

    # Gets the config files' names from the arguments
    if len(sys.argv) > 1:
        configs = sys.argv[1:]
        for config in configs:
            if not os.path.isfile(config):
                # Configuration file doesn't exist
                print("Configuration file {0} not found. ".format(config))
                sys.exit()
    else:
        fallBackConfig = "testconfig.txt"
        if os.path.isfile(fallBackConfig):
            print("Using demo configuration..")
            configs.append(fallBackConfig)
        else:
            print("No configuration file provided. Exiting..")
            sys.exit()

    print("Configuration file(s): {0}".format(configs))
    infodensRun(configs)
