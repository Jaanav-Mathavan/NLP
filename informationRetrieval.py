from util import *
from models.tfidf import TFIDF
from models.lsa import LSA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from models.sentence_tranformer import SentenceTransformerEmbedder

class InformationRetrieval():

    def __init__(self, model = "tfidf", qex = False, dex = False, include_bigrams = False):
        self.index = None
        self.model = model
        self.qex = qex
        self.dex = dex
        self.include_bigrams = include_bigrams
        self.lsa_model = LSA()
        self.use_em = True if "embeddings" in self.model else False
        self.use_lsa = True if "lsa" in self.model else False
        self.use_wordnet = True if "wordnet" in self.model else False
        if self.use_wordnet:
            self.tfidf_model = TFIDF(self.qex, self.dex, self.include_bigrams, use_wn=True) 
        else:
            self.tfidf_model = TFIDF(self.qex, self.dex, self.include_bigrams) 
        self.docIDs = None
        if self.use_em:
            self.st_model = SentenceTransformerEmbedder()
        self.docID_to_idx = {}
        self.idx_to_docID = {}
            
    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """
        self.docIDs = docIDs
        self.docID_to_idx = {doc_id: idx for idx, doc_id in enumerate(docIDs)}
        self.idx_to_docID = {idx: doc_id for idx, doc_id in enumerate(docIDs)}
        
        flattened_docs = []
        index = {}

        for doc, doc_id in zip(docs, docIDs):
            words = [word for sentence in doc for word in sentence]
            flattened_docs.append(" ".join(words))
            for word in words:
                if word not in index:
                    index[word] = set()
                index[word].add(doc_id)
        
        for word in index:
            index[word] = list(index[word])
        self.index = index
            
        self.compute_tf_idf_matrix(flattened_docs)
        
        if self.use_lsa:
            self.compute_lsa_matrix(self.tfidf_model.get_doc_matrix())
            
        if self.use_em:
            self.compute_st_matrix(flattened_docs)

    def compute_tf_idf_matrix(self, flattened_docs, docIDs=None):
        """
        Computes the tf-idf matrix for the documents

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a document and
            each sub-sub-list is a sentence of the document

        Returns
        -------
        list
            A list of lists of floats where the ith sub-list is a list of tf-idf
            values for the ith document
        """
        self.tfidf_model.fit(flattened_docs)
    
    def compute_lsa_matrix(self, doc_matrix):
        """
        Computes the LSA matrix for the documents

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a document and
            each sub-sub-list is a sentence of the document

        Returns
        -------
        list
            A list of lists of floats where the ith sub-list is a list of LSA
            values for the ith document
        """
        self.lsa_model.fit(doc_matrix)
        
    def compute_st_matrix(self, flattened_docs):
        """
        Computes the Word2Vec matrix for the documents

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a document and
            each sub-sub-list is a sentence of the document

        Returns
        -------
        list
            A list of lists of floats where the ith sub-list is a list of Word2Vec
            values for the ith document
        """
        self.st_model.fit(flattened_docs)
        
    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query

        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """
        doc_IDs_ordered = []

        for query in queries:
            words = [word for sentence in query for word in sentence if word in self.index]
            query_text = " ".join(words)
            if self.use_em:
                query_text = " ".join([word for sentence in query for word in sentence])
                query_vector = self.st_model.transform([query_text])
                similarity_scores = cosine_similarity(self.st_model.get_doc_matrix(), query_vector).flatten()    
            elif self.use_lsa:
                query_vector = self.lsa_model.transform(query_vector)
                similarity_scores = cosine_similarity(self.lsa_model.get_doc_matrix(), query_vector).flatten()
            else:
                query_vector = self.tfidf_model.transform([query_text])
                similarity_scores = cosine_similarity(self.tfidf_model.get_doc_matrix(), query_vector).flatten()

            relevant_docs = set()
            for word in words:
                if word in self.index:
                    relevant_docs.update(self.index[word])
            
            if relevant_docs:
                relevant_indices = [self.docID_to_idx[doc_id] for doc_id in relevant_docs]
                relevant_scores = [(self.idx_to_docID[idx], similarity_scores[idx]) for idx in relevant_indices]
                ranked_docs = [str(doc_id) for doc_id, _ in sorted(relevant_scores, key=lambda x: x[1], reverse=True)]
            else:
                ranked_docs = []

            doc_IDs_ordered.append(ranked_docs)

        return doc_IDs_ordered

    
#from util import *
#import numpy as np
## Add your import statements here
#
#class InformationRetrieval():
#
#	def __init__(self):
#		self.index = None
#
#	def buildIndex(self, docs, docIDs):
#		"""
#		Builds the document index in terms of the document
#		IDs and stores it in the 'index' class variable
#
#		Parameters
#		----------
#		arg1 : list
#			A list of lists of lists where each sub-list is
#			a document and each sub-sub-list is a sentence of the document
#		arg2 : list
#			A list of integers denoting IDs of the documents
#		Returns
#		-------
#		None
#		"""
#		self.docs = docs 
#		self.docIDs = docIDs
#		index = {}
#		for doc, id in zip(docs, docIDs):
#			for segment in doc: 
#				for word in segment: 
#					if word not in index:
#						index[word] = set()
#					index[word].add(id)
#		for word in index:
#			index[word] = list(index[word])
#		self.index = index
#		self.compute_tf_idf_matrix(self.docs, self.docIDs)
#	def compute_tf_idf_matrix(self, docs, docIDs):
#		"""
#		Computes the tf-idf matrix for the documents
#
#		Parameters
#		----------
#		arg1 : list
#			A list of lists of lists where each sub-list is a document and
#			each sub-sub-list is a sentence of the document
#
#		Returns
#		-------
#		list
#			A list of lists of floats where the ith sub-list is a list of tf-idf
#			values for the ith document
#		"""
#		self.words = list(self.index.keys())
#		self.tf_idf_matrix = [[0]*len(docs) for _ in range(len(self.words))] 
#		self.idf_list = []
#		for i in range(len(self.words)):
#			for j, doc in zip(docIDs, docs):
#				doc_words = [word for segment in doc for word in segment]
#				word_count = len(list(filter(lambda x: x == self.words[i], doc_words)))
#				tf = word_count / len(doc_words) if len(doc_words) > 0 else 0
#				idf = np.log(len(docs) / len(self.index[self.words[i]]))
#				self.tf_idf_matrix[i][j-1] = tf * idf
#				self.idf_list.append(idf)
#
#	def rank(self, queries):
#		"""
#		Rank the documents according to relevance for each query
#
#		Parameters
#		----------
#		arg1 : list
#			A list of lists of lists where each sub-list is a query and
#			each sub-sub-list is a sentence of the query
#		
#
#		Returns
#		-------
#		list
#			A list of lists of integers where the ith sub-list is a list of IDs
#			of documents in their predicted order of relevance to the ith query
#		"""
#		doc_IDs_ordered = []
#		tf_idf_query = []
#		for query in queries: 
#			query_vector = np.zeros(len(self.words))
#			query_words = []
#			query_docs = []
#			for sentence in query: 
#				for word in sentence:	
#					if word not in self.index:
#						continue
#					query_docs += self.index[word]
#					query_words.append(word)
#			query_docs = list(set(query_docs))
#			for i in range(len(self.words)):
#				word_count = len(list(filter(lambda x: x == self.words[i], query_words)))
#				tf = word_count / len(query_words) if len(query_words) > 0 else 0
#				query_vector[i] = tf * self.idf_list[i]
#   
#			similarities = self.cosine_similarity_matrix(self.tf_idf_matrix, np.array(query_vector).T, query_docs)
#			doc_IDs_ordered.append([str(doc_id) for doc_id, _ in sorted(similarities, key=lambda x: x[1], reverse=True)])
#   
#		
#		return doc_IDs_ordered
#
#
#	def cosine_similarity_matrix(self, doc_matrix, query_vector, query_docs):
#		doc_matrix = np.array(doc_matrix)
#		query_norm = np.linalg.norm(query_vector)
#		query_norm = query_norm if query_norm != 0 else 1e-10
#		results = []
#		for doc_id in query_docs:
#			doc_vec = doc_matrix[:, doc_id-1]
#			doc_norm = np.linalg.norm(doc_vec)
#			doc_norm = doc_norm if doc_norm != 0 else 1e-10
#			similarity = (doc_vec @ query_vector) / (doc_norm * query_norm)
#			results.append((doc_id, similarity.item()))
#		return results

#Comparing code using official tf-idf vectorizer. The above is coded from scratch.
