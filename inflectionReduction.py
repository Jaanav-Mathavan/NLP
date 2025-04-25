from util import *
import nltk
from nltk.stem import PorterStemmer
# Add your import statements here




class InflectionReduction:
	def __init__(self):
		self.porter_stemmer = PorterStemmer() #Initiating the Porter Stemmer Model
	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = None
		reducedText = []
		for tokens in text:
			curr_token_list = []
			for token in tokens:
				#Looped through every token in list of list of words after word tokenization and performed stemming. Lemmatization using wordNet will be done in 1B.
				curr_token_list.append(self.porter_stemmer.stem(token))
			reducedText.append(curr_token_list)
		#Fill in code here
		return reducedText


