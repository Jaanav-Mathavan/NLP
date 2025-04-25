from util import *
import nltk as nltk 

# Add your import statements here

class SentenceSegmentation():
    def __init__(self):
        #Exceptions for Abbreviations
        self.exceptions = ["Mr", "Mrs", "Ms", "Dr", "Prof", "Inc", "no", "rev", "Ltd" "e.g", "i.e", "a.m", "p.m"]
        #Numbers such as 1., 2., 3. should also be considered as abbreviations in order to prevent being seperated
        self.exceptions += [str(i) for i in range(10)] 
        #initiating sentence tokenizer
        self.sentence_tokenizer = nltk.tokenize.sent_tokenize 
        #Punctuations : .?!;
        self.punctuations = ".?!;"
    def naive(self, text): # Top-Down Approach - Classical AI - Deduction
        """
        Sentence Segmentation using a Naive Approach

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """
        segmentedText = None 
        segmentedText = []
        words = text.split(" ")
        temp_list = []
        for word in words:
            if word == "":
                continue
            #Check for Abbreviations.
            if any(exception in word for exception in self.exceptions):
                #If there is any abbreviations that ends with a ] being the last word in the sentence, this will help to detect. E.g.:[Time now is 10 a.m.].
                if any(char in word for char in ")}]"):
                    for bracket in ")}]":
                        if bracket in word:
                            temp_list.append(word[:word.index(bracket)+1])
                            break
                    #And look for punctuation ending
                    if any(char == word[-1] for char in self.punctuations):
                        segmentedText.append(" ".join(temp_list))
                        temp_list = [] 
                else:
                    temp_list.append(word)
            else:
                #If last word is from any of the punctuations given above
                if any(char == word[-1] for char in self.punctuations):
                    temp_list.append(word)
                    segmentedText.append(" ".join(temp_list))
                    temp_list = [] 
                else:
                    temp_list.append(word)  
        #Fill in code here

        return segmentedText

    
    def punkt(self, text): # Bottom-Up Approach - ML - Induction
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each strin is a single sentence
        """

        segmentedText = None
        segmentedText = self.sentence_tokenizer(text)
        #Fill in code here
        
        return segmentedText