
# INFODENS

This framework provides a quick way to generate hand-crafted and learned features from text and compare their performance. It is also designed to expedite feature engineering tasks.


## Setup & Requirements

The framework is written entirely in Python (3.x supported) so it runs without compilation. However, you still need to install the required dependencies. If you are using Windows, you might find it easier to install a Python distribution like Anaconda or Canopy.

The following libraries are required:

* [NumPy](http://www.numpy.org/) & [SciPy](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/install.html)
* [Keras](https://keras.io/#installation) with backend (TensorFlow recommended).
* [NLTK](https://www.nltk.org/install.html) with datasets.
* [configparser](https://docs.python.org/3/library/configparser.html)
* [joblib](https://pythonhosted.org/joblib/installing.html)
* [gensim](https://radimrehurek.com/gensim/) To generate word Embeddings.
* [fastText](https://github.com/facebookresearch/fastText) for Python. For fastText features.
* [liac-arff](https://pypi.python.org/pypi/liac-arff) To output features in the arff format.

If you use ngram language model features, download and compile one of the following:
* [KenLM](https://github.com/kpu/kenlm) with Python module.
* [SRILM](http://www.speech.sri.com/projects/srilm/)

## Running the toolkit

The toolkit takes a configuration file (INI format) as an input in which all the required parameters are specified.

To run it:

```
python infodens.py democonfig.txt
```

The mandatory parameters for the config file are:

```
[Input]
train file : trainFile
train classes : classesFile
test file : testFile
test classes: testClassesFile

[Features]
featId1 : argString1
featId2 : argString2
.
.
featIdN : argStringN
```
where "trainFile" is the name of the file containing the sentences (one sentence per line) for which the training features will be extracted and used to train a classifier/regressor. The "classesFile" is the classes file with each line specifying the class label for the corresponding sentence in the "trainFile".

Similar ot the train files, the "testFile" and "testClassesFile" are the test files used for evaluation of the features and classifier.

The required features are called by their IDs and after a colon or equals sign the arguments of that feature are specified if needed.

The current supported features are described in the table at the bottom of the page.

All the config file parameters in their respective sections are shown below with description:

```
[Input]

# Functions as above
train file : sentences.train

# Specifies the path for the file containing the train class labels/values.
# Each line gives the label to the corresponding input sentence
train classes:  data/testSentClasses2.txt

# Those two files need to be both specified for a test run.
test file : sentences.test
test classes: classesFile.test 

# Auxiliary feature files can be used to add to the extracted features. 
# Specify the paths for the libsvm files seprated by space.
train feats: feats_trainFeats.libsvm feats2_trainFeats.libsvm
test feats: feats_testFeats.libsvm feats2_testFeats.libsvm

# Here you provide the corpus to be used for building language models and word embeddings
training corpus: data/testSent2.txt

# ISO 639 code of the language of the files
language : eng

[Settings]

# Provide the path for SRILM or KenLM here for the language model features
srilm : srilm/bin
kenlm: kenlm/bin

# The maximum number of processes to run
threads : 4

[Output]

# To persist the models simply give a file name to the persist 
# parameter. The names of the classifiers will be preappended.
persist model: testClass.skl

# The classification report output file name
classifier report: report1.txt

# feature output file name followed by format(s) (libsvm, csv, and arff supported)
output features: feats_out libsvm csv

[Features]
# As described above
1
2
4: -ngram 1 -cutoff 2
4: -ngram 2 -cutoff 5

[Classifiers]

# Provide the names of classifiers required each on a line (case sensitive)
# Available classifiers : Decision_tree, Random_forest, Ada_boost, Ensemble
# The SVR_linear regressor is also available for regression
SVR_linear

# Classifiers also take argument strings
# A default -rank argument implements feature ranking
# Follow it with N for only the top N ranking features and -1 for ranking all
SVC_linear: -rank 5


```

## Multilingual runs

The toolkit can be run on multilingual (or parallel) corpora for tasks like Machine translation quality estimation. To do so, provide multiple configuration files as arguments, where each file represents one run of the toolkit with the input files and the required features. The classification/regression parameters are collected from all input configuration files and used on the merged feature output from all runs.

Example:

```
python infodens.py democonfig.txt config2.txt
```

## Predict runs

The toolkit can also be used to give labels for unseen data. To do that, simply replace the test parameters (file and classes) with the "predict file" as below. The label files from different models will be preappended with the model's name. The train files must also be given.

Example:

```
# If only prediction is required then provide this parameter instead of the test ones.
# It specifies the sentences to generate labels/values for
predict file : sentences.test
```

## Developer's guide

The tool is mainly designed to ease the tasks of feature engineering. The Wiki contains a simple guide to help researchers code their own features. We also hope to encourage researchers and developers to adapt the code to their needs, for example even change the preprocessor and configurator and use the skeleton of the toolkit for other machine learning tasks.
We welcome (and encourage) any feedback and inquiries.

## Citation

An overview of the framework's architecture along with a detailed evaluation is available in [this technical paper](https://arxiv.org/abs/1810.07091). If you use the framework in your work, please cite:

```
@article{2018infodens,
  author = {Ahmad Taie and Raphael Rubino and Josef van Genabith},
  title = "{INFODENS: An Open-source Framework for Learning Text Representations}",
  journal = {arXiv preprint arXiv:1810.07091},
  year      = {2018},
  eprint = {1810.07091}
}
```


## List of Features:

Feature Name | ID | Description | Argument string
--- | --- | --- | ---  
Average word length | 1 | Calculates the average word length per sentence | None
Syllable ratio | 2 | Counting the number of vowel-sequences that are delimited by consonants or space in a word, normalized by the number of tokens in the sentence | Example: a,e,i,o,u  <li> List of vowels (comma separated). Default: lowercase English vowels </li>
Sentence length | 10 | Calculates the length of each sentence in words | None
Lexical density | 3 | The frequency of tokens that are not tagged with the POS tags given. Computed by dividing the number of tokens not tagged with the given POS tags by the number of tokens in the sentence. (Uses NLTK tags by default) | Mandatory parameter: </li> <li> -pos_tags : list of POS tags (comma separated) </li> The following are optional parameters: <li> -pos_train : Path for POS tagged train sentences. </li>  <li> -pos_test : Path for POS tagged test sentences.
Lexical to tokens ratio | 12 | The ratio of lexical words (given POS tags) to tokens in the sentence | Mandatory parameter: </li> <li> -pos_tags : list of POS tags (comma separated) </li> The following are optional parameters: <li> -pos_train : Path for POS tagged train sentences. </li>  <li> -pos_test : Path for POS tagged test sentences.
Lexical richness (type-token ratio) | 11 | The ratio of unique tokens in the sentence over the sentence length | None 
Ngram bag of words | 4 | Ngram bag of words | Optional parameters: <li> -ngram : Order of ngram. (Default = 1: Unigrams) </li> <li> -cutoff : Minimum cutoff frequency for ngram. (Default = 1) </li> <li> -hash_size : Length of output vector for the hashing trick. (Default = None, hashing trick not used.) </li>  <li> -train : Path for file to build ngram vector from. (Default is train sentences) </li> 
Ngram bag of POS | 5 | Ngram bag of POS |  Optional parameters: <li> -ngram : Order of ngram. (Default = 1: Unigrams) </li> <li> -cutoff : Minimum cutoff frequency for ngram. (Default = 1) </li> <li> -hash_size : Length of output vector for the hashing trick. (Default = None, hashing trick not used.) </li> <li> -proc_train : Path for POS tagged train sentences. </li> <li> -proc_test : Path for POS tagged test sentences. </li> <li> -train : Path for file to build ngram vector from. (Default is train sentences) </li> 
Ngram bag of mixed words | 6 | Ngram bag of mixed words, sentences are tagged and only tags that start with J,N,V, or R are left, the others are actual words (Tagged with NLTK) | Optional parameters: <li> -ngram : Order of ngram. (Default = 1: Unigrams) </li> <li> -cutoff : Minimum cutoff frequency for ngram. (Default = 1) </li> <li> -hash_size : Length of output vector for the hashing trick. (Default = None, hashing trick not used.) </li>  <li> -proc_train : Path for mixed train sentences. </li> <li> -proc_test : Path for mixed test sentences. </li> <li> -train : Path for file to build ngram vector from. (Default is train sentences) </li> 
Ngram bag of lemmas | 7 | Ngram bag of lemmas (lemmatized using NLTK WordNetLemmatizer) | Optional parameters: <li> -ngram : Order of ngram. (Default = 1: Unigrams) </li> <li> -cutoff : Minimum cutoff frequency for ngram. (Default = 1) </li> <li> -hash_size : Length of output vector for the hashing trick. (Default = None, hashing trick not used.) </li> <li> -proc_train : Path for lemmatized train sentences. </li> <li> -proc_test : Path for lemmatized test sentences. </li> <li> -train : Path for file to build ngram vector from. (Default is train sentences) </li> 
Perplexity language model | 17 | Using provided tool to build a language model (or use the one given) then compute the sentence scores (log probabilities) and perplexities. | Optional parameters: <li> -lm : Given language model </li> <li> -ngram : Ngram of the LM and feature (default=3) </li>
Perplexity language model POS | 18 | Using provided tool to build a language model (or use given one) then compute the sentence scores (log probabilities) and perplexities for POS tagged sentences | Optional parameters:  <li> -ngram : Ngram of the LM and feature (default=3) </li>  <li> -pos_train : Path for POS tagged train sentences. </li>  <li> -pos_test : Path for POS tagged test sentences. </li>  <li> -pos_corpus :  Path for POS tagged corpus. </li>  <li> -pos_lm :  Path for POS language model. </li>
Surprisal log probability | 20 | Using provided tool to build a language model (or use given one) then compute the sentence scores in units of bits (log2 probabilities) and perplexities. | Optional parameters: <li> -lm : Given language model </li> <li> -ngram : Ngram of the LM and feature (default=3) </li>
Surprisal POS log probability | 21 | Using provided tool to build a language model (or use given one) then compute the sentence scores in units of bits (log2 probabilities) and perplexities for POS tagged sentences (Default: NLTK tagger) | Optional parameters:  <li> -ngram : Ngram of the LM and feature (default=3) </li>  <li> -pos_train : Path for POS tagged train sentences. </li>  <li> -pos_test : Path for POS tagged test sentences. </li>  <li> -pos_corpus :  Path for POS tagged corpus. </li>  <li> -pos_lm :  Path for POS language model. </li>
Ngram frequency quantile distribution | 19 | Models the input sequence as a frequency distribution over quantiles | <li> -ngram : Order of ngram. (Default = 1: Unigrams) </li> <li> -cutoff : Minimum cutoff frequency for ngram. (Default = 1) </li> <li> -n_quantiles : number of quantiles (default=4) </li> 
Sentence log probability variance | 77 | Using provided tool to build a language model (or use given one) then compute variance of incremental per-word sentence scores. | Optional parameters: <li> -lm : Given language model </li> <li> -ngram : Ngram of the LM and feature (default=3) </li>
Word vector average | 33 | Trains or uses a word2vec model (gensim) and gets the average of all word vectors per sentence | Example: 200 <li> Vector length (default 100) </li> or: models\vecModel.ml <li> Path to the word embeddings model </li>
FastText learned embeddings | 801 | Uses the learned sentence embedding from training a fastText model on the train data and classes. | Optional parameters: <li> -epochs  number of training epochs (default = 10) </li> <li> -dim: Length of embeddings vector. (default=100) </li> <li> -wordNgrams : Ngram features to add. (default=0) </li> <li> -lr : learning rate. (default=1.0) </li>


# B6-SFB1102
This toolkit is part of the B6 project of SFB1102 -- http://www.sfb1102.uni-saarland.de

