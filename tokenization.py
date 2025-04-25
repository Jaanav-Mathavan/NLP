from util import *
import nltk 
from nltk.tokenize.treebank import TreebankWordTokenizer
# Add your import statements here




class Tokenization():
	def __init__(self):
		#If I import nltk.tokenize.word_tokenize then it includes TreebankWordTokenizer along with PunktSentenceTokenizer. So I am specifically using TreebankWordTokenizer
		self.word_tokenizer = TreebankWordTokenizer()
  		#Exceptions for Abbreviations
		self.exceptions = ["Mr", "Mrs", "Ms", "Dr", "Prof", "Inc", "no", "rev", "Ltd" "e.g", "i.e", "a.m", "p.m"]
		#Numbers such as 1., 2., 3. should also be considered as abbreviations in order to prevent being seperated
		self.exceptions += [str(i) for i in range(10)] 
  		#Unlike sentences here even brackets need to be considered a separate token
		self.punctuations = ".,!?;:'\"()[]{}" 
	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None
		tokenizedText = []
		for segment in text:
			tokens = []
			current_word = ""
			for char in segment:
       			#Check for alpha numebric characters or apostrophe
				if char.isalnum() or char == "'":
					current_word += char 
				else:
					#Decimal places, exceptions based words
					if current_word and current_word in self.exceptions and char == ".":
						current_word += char
					if current_word:
						tokens.append(current_word)
						current_word = "" 
					#Append to tokens seperately if char is one of the punctuations. We arent removing any punctuations for now.
					if char in self.punctuations or char in "+-/*^&@#%^":
						tokens.append(char)
			tokenizedText.append(tokens)
					
		#Fill in code here

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		tokenizedText = None
		tokenizedText = []
		for segment in text:
			tokenizedText.append(self.word_tokenizer.tokenize(segment))
		#Fill in code here

		return tokenizedText