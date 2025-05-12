from util import *
import re
from nltk.tokenize.treebank import TreebankWordTokenizer

class Tokenization:
    def naive(self, text):
        """
        Tokenization using a Naive Approach, removing punctuation

        Parameters
        ----------
        text : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of alphanumeric tokens
        """
        tokenizedText = []
        for sentence in text:
            # Match only alphanumeric words, exclude punctuation
            tokens = [token.lower() for token in re.findall(r'\b[a-zA-Z0-9]+\b', sentence.lower())]
            tokenizedText.append(tokens)
        return tokenizedText

    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer, removing punctuation

        Parameters
        ----------
        text : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of alphanumeric tokens
        """
        tokenizedText = []
        tokenizer = TreebankWordTokenizer()
        for sentence in text:
            # Filter out non-alphanumeric tokens
            tokens = [token.lower() for token in tokenizer.tokenize(sentence) if token.isalnum()]
            tokenizedText.append(tokens)
        return tokenizedText
