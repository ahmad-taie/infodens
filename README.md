
# INFODENS

This toolkit provides a quick way to generate features from text, and expedites feature engineering tasks.


# Setup & Requirements

The tool is written entirely in Python (2.x or 3.x supported) so it runs without compilation. However, you still need to install the required dependencies which are listed in the Wiki. If you are using Windows, you might find it easier to install a Python distribution like Anaconda or Canopy.

# Running the toolkit

The toolkit takes a configuration file as an input in which all the required parameters are specified.

To run it:

```
python infodens.py democonfig.txt
```

The mandatory parameters for the config file are:

```
input files : inputText

featId1 argString1
featId2 argString2
.
.
featIdN argStringN
```
where "inputText" is the name of the file containing the sentences (one sentence per line) for which the features will be generated.

The required features are then called by their IDs and after a white space the arguments of that feature are specified if needed.

The current supported features are described in the table below.

Optional parameters are shown below with description:

```
# Specifies the path for the file containing the class labels.
# Each line gives the label to the corresponding input sentence
# This parameter has to be specified for classification, and feature output
input classes:  data/testSentClasses2.txt

# Here you provide the corpus to be used for building language models and word embeddings
training corpus: data/testSent2.txt

# If SRILM is prefered for building language models, provide the binaries' path here
srilm path : srilm/bin

# ISO 639 code of the language of the files
operating language : eng

# The maximum number of processes to run
threads : 3

# How many folds of Cross validation
folds : 1

# Provide the classifiers required seprated by space
#classifiers : Decision_Tree Random_Forest Ada_Boost Ensemble
classifiers : SVC_linear

# The classification report output
output classifier: report1.txt

# feature output file and format (libsvm and arff supported)
output features: feats.txt libsvm

```

# Multilingual runs

The toolkit can be run on multilingual (or parallel) corpora for tasks like Machine translation quality estimation. To do so, provide multiple configuration files as arguments, where each file represents one run of the toolkit with the input files and the required features. The classification/regression parameters are collected from all input configuration files and used on the merged feature output from all runs.

Example:

```
python infodens.py democonfig.txt config2.txt
```

# Developer's guide

The tool is mainly designed to ease the tasks of feature engineering. The Wiki contains a simple guide to help researchers code their own features. We also hope to encourage researchers and developers to adapt the code to their needs, for example even change the preprocessor and configurator and use the skeleton of the toolkit for other machine learning tasks.
We welcome (and encourage) any feedback and inquiries.


# List of Features:

Feature Name | ID | Description | Argument string
--- | --- | --- | ---  
Average word length | 1 | Calculates the average word length per sentence | None
Syllable ratio | 2 | Counting the number of vowel-sequences that are delimited by consonants or space in a word, normalized by the number of tokens in the sentence | None 
Sentence length | 10 | Calculates the length of each sentence in words | None
Lexical density | 3 | The frequency of tokens that are not nouns, adjectives, adverbs or verbs. Computed by dividing the number of tokens not tagged with the given POS tags by the number of tokens in the sentence |  Example: 1,NN, c:\tagged.txt <li> Flag (0/1) indicating given POS tagged input. </li> <li> List of POS tags (comma separated) </li> <li> POS tagged file path (when flag is 1) </li>
Lexical richness (type-token ratio) | 11 | The ratio of unique tokens in the sentence over the sentence length | None 
Lexical to tokens ratio | 12 | The ratio of lexical words (given POS tags) to tokens in the sentence | Example: 1,NN,c:\tagged.txt <li> Flag (0/1) indicating given POS tagged input. </li> <li> List of POS tags (comma separated) </li> <li> POS tagged file path (when flag is 1) </li>
Ngram bag of words | 4 | Ngram bag of words | Example: 1,2  (Uni-grams that appear at least twice) <li> N in ngram </li> <li> Cutoff frequency </li> 
Ngram bag of POS | 5 | Ngram bag of POS | Example: 1,2 or 1,2,c:\taggedInput.txt  (Uni-grams that appear at least twice) <li> N in ngram </li> <li> Cutoff frequency </li> <li> Tagged POS input (Optional) </li>
Ngram bag of mixed words | 6 | Ngram bag of mixed words, sentences are tagged and only tags that start with J,N,V, or R are left, the others are actual words (Tagged with NLTK) | Example: 1,2 (Uni-grams that appear at least twice)  <li> N in ngram </li> <li> Cutoff frequency </li> <li> Tagged POS input (Optional) </li>
Ngram bag of lemmas | 7 | Ngram bag of lemmas (lemmatized using NLTK WordNetLemmatizer) | Example: 1,2 (Uni-grams that appear at least twice) <li> N in ngram </li> <li> Cutoff frequency </li> <li> Tagged POS input (Optional) </li>
Perplexity language model | 17 | Using SRILM's ngram or KenLM to build a language model then compute the sentence scores (log probabilities) and perplexities. | Example: 0,3 (Trigrams language model) or 1,3,models\myLM.lm <li> Flag (0/1) Given language model </li> <li> Ngram of the LM and feature </li> <li> Language model file path (if flag is 1) </li>
Perplexity language model POS | 18 | Using SRILM's ngram or KenLM to build a language model then compute the sentence scores (log probabilities) and perplexities for POS tagged sentences | Example: 0,1,0,3,models\myLM.lm <li> Tagged input flag (1/0) </li> <li> Given LM flag (0/1) </li> <li> Given tagged Corpus flag (0/1) </li> <li> Ngram order of model and feature </li> <li> Tagged POS file path (if Tagged inp. flag) </li> <li> LM file Path (if LM flag) </li> <li> Tagged Corpus path (if no LM flag but Tagged Corpus flag) </li>
Surprisal log probability | 20 | Using SRILM's ngram or KenLM to build a language model then compute the sentence scores in units of bits (log2 probabilities) and perplexities. | Example: 0,3 (Trigrams language model) or 1,3,models\myLM.lm <li> Flag (0/1) Given language model </li> <li> Ngram of the LM and feature </li> <li> Language model file path (if flag is 1) </li>
Surprisal POS log probability | 21 | Using SRILM's ngram or KenLM to build a language model then compute the sentence scores in units of bits (log2 probabilities) and perplexities for POS tagged sentences | Example: 0,1,0,3,models\myLM.lm <li> Tagged input flag (1/0) </li> <li> Given LM flag (0/1) </li> <li> Given tagged Corpus flag (0/1) </li> <li> Ngram order of model and feature </li> <li> Tagged POS file path (if Tagged inp. flag) </li> <li> LM file Path (if LM flag) </li> <li> Tagged Corpus path (if no LM flag but Tagged Corpus flag) </li>
Ngram frequency quantile distribution | 19 | Models the input sequence as a frequency distribution over quantiles | Example: 1,1,4 <li> N in ngram </li> <li> Cutoff frequency </li> <li> Number of quantiles </li>
Word vector average | 33 | Trains or uses a word2vec model (gensim) and gets the average of all word vectors per sentence | Example: 200 <li> Vector length (default 100) </li> or: models\vecModel.ml <li> Path to the word embeddings model </li>


# B6-SFB1102
This toolkit is part of the B6 project of SFB1102 -- http://www.sfb1102.uni-saarland.de

