from util import *
import nltk 
from nltk.corpus import stopwords
# Add your import statements here




class StopwordRemoval():
	def __init__(self):
     
		self.stopwords = stopwords.words("english") #NLTK predefined stop words
  
	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = None
		stopwordRemovedText = []
		#We use nltk stopwards dictionary to get all possible stopwords and elimnate them from each stemmed list. We can also build a corpus specific stopword for which I built a function in the util.py file and it is called after stemming in the main.py file
		for reduced_tokens in text:
			reduced_tokens_filtered = [t for t in reduced_tokens if t.lower() not in self.stopwords]
			stopwordRemovedText.append(reduced_tokens_filtered)
		#Fill in code here

		return stopwordRemovedText




	