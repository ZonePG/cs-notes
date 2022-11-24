# Language Representations and RNNs

## Word Representations

### Vector space models of words

While learning these word representations, we are actually building a vector space in which all words reside with certain relationships between them

Encodes both syntactic and semantic relationships

This vector space allows for algebraic operations:

Vec(king) – vec(man) + vec(woman) ≈ vec(queen)

### Word Representation Resources

Word-level representations
- Word2Vec (Google, 2013) 
  - https://code.google.com/archive/p/word2vec/
- Glove (Stanford, 2014)
  - https://nlp.stanford.edu/projects/glove/
- FastText (Facebook, 2017)
  - https://fasttext.cc/

Sentence-level representations
- ELMO (Allen Institute for AI, 2018)
  - https://allennlp.org/elmo
- BERT (Google, 2018)
  - https://github.com/google-research/bert
- RoBERTa (Facebook, 2019)
  - https://github.com/pytorch/fairseq

## Sentence Modeling

- Sequence Label Prediction
- Sequence Prediction
  - Main Challenges:
  - Sequences of variable lengths (e.g., sentences)
  - Keep the number of parameters at a minimum
  - Take advantage of possible redundancy
- Language Model: predict next words
  - Language Generation
  - Image Caption
  - Speech Recognition

## Recurrent Neural Network

## Gated Recurrent Neural Network

LSTM

ELMO: Two bi-directional LSTMs are used to contextualize the word embeddings https://allennlp.org/elmo

## Syntax and Language Structure

## Recursive Neural Network

### Tree-based RNNs (or Recursive Neural Network)

- Recursive Neural Network for Sentiment Analysis
  - Socher et al., Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank, EMNLP 2013
- Stack LSTM
  - Dyer et al., Transition-Based Dependency Parsing with Stack Long Short-Term Memory, 2015

### Resources

- Stanford NLP software
  - https://nlp.stanford.edu/software/
  - Stanford Parser
  - Stanford POS Tagger
- UC Berkeley Parser
  - https://github.com/slavpetrov/berkeleyparser
- Parsers by Kenji Sagae (syntactic parsers)
  - http://www.sagae.org/software.html

