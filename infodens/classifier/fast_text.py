from infodens.classifier.classifier import Classifier
import fasttext
import os


class Fast_text(Classifier):

    classifierName = 'Fast_text'

    def dump_labelize(self, sents, labls, mergedFile):

        if labls:
            for i in range(0, len(sents)):
                sents[i] = "__label__{0} {1}".format(labls[i], sents[i])

        with open(mergedFile, "w") as file:
            file.writelines(sents)

    def train(self):
        outFile = "ft_runTrain.txt"
        self.dump_labelize(self.Xtrain, self.ytrain, outFile)

        # Save model in class variable
        self.model = fasttext.supervised(input_file=outFile, output='model_ft_', epoch=100,
                                         word_ngrams=1)
        os.remove(outFile)

    def predict(self):
        labels = self.model.predict(self.Xtest)
        return [float(lab[0]) for lab in labels]


