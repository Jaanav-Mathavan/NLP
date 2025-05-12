from util import *
from nltk.stem import PorterStemmer

class InflectionReduction:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def reduce(self, text):
        """
        Stem each token in the input text using PorterStemmer

        Parameters
        ----------
        text : list
            A list of lists where each sub-list is a sentence of tokens

        Returns
        -------
        list
            A list of lists with stemmed tokens
        """
        return [[self.stemmer.stem(token) for token in sentence] for sentence in text]
