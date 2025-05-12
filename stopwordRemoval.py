from util import *
from nltk.corpus import stopwords
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class StopwordRemoval:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))

    def fromList(self, text):
        """
        Remove stopwords and non-alphanumeric tokens from the input text

        Parameters
        ----------
        text : list
            A list of lists where each sub-list is a sentence of tokens

        Returns
        -------
        list
            A list of lists with stopwords and non-alphanumeric tokens removed
        """
        return [[token for token in sentence if token.isalnum() and token.lower() not in self.stopwords] for sentence in text]	
