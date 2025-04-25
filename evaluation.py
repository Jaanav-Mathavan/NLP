from util import *
import numpy as np
# Add your import statements here

class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		precision = len(list(filter(lambda x: x in true_doc_IDs, query_doc_IDs_ordered[:k]))) / k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		precision_list = []
		for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
			query_id = str(query_id)
			rel_doc_ids = [x["id"] for x in qrels if x["query_num"] == query_id]
			q_precision = self.queryPrecision(doc_id_order, query_id, rel_doc_ids, k)
			precision_list.append(q_precision)
		meanPrecision = sum(precision_list) / len(precision_list) if precision_list else 0.0
		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		recall = len(list(filter(lambda x: x in true_doc_IDs, query_doc_IDs_ordered[:k]))) / len(true_doc_IDs) if true_doc_IDs else 0.0
  
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""
		meanRecall = -1

		#Fill in code here
		recall_list = []
		for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
			query_id = str(query_id)
			rel_doc_ids = [x["id"] for x in qrels if x["query_num"] == query_id]
			q_recall = self.queryRecall(doc_id_order, query_id, rel_doc_ids, k)
			recall_list.append(q_recall)
		meanRecall = sum(recall_list) / len(recall_list) if recall_list else 0.0

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1
		alpha = 0.5
		#Fill in code here
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		fscore = (1 + alpha**2) * (precision * recall) / ((alpha **2 * precision) + recall) if (precision + recall) > 0 else 0.0 #alpha = 0.5 & precision, recall != 0
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		fscore_list = []
		for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
			query_id = str(query_id)
			rel_doc_ids = [x["id"] for x in qrels if x["query_num"] == query_id]
			q_fscore = self.queryFscore(doc_id_order, query_id, rel_doc_ids, k)
			fscore_list.append(q_fscore)
		meanFscore = sum(fscore_list) / len(fscore_list) if fscore_list else 0.0
		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query
		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			A list of dictionaries containing document-relevance
			[INCORRECT] The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value
		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1
		#Fill in code here
		true_docs = {x["id"]: 5 - x["position"] for x in qrels if x["query_num"] == query_id}
		dcg = 0
		for index, retrieved_doc in enumerate(query_doc_IDs_ordered[:k]):
			if retrieved_doc in true_docs:
				dcg += true_docs[retrieved_doc] / np.log2(index + 2)
		# Calculating ideal DCG
		ideal_dcg = 0
		for index, true_doc in enumerate(sorted(true_docs.keys(), key=lambda x: true_docs[x], reverse=True)[:k]):
			ideal_dcg += true_docs[true_doc] / np.log2(index + 2)
		# Normalizing DCG
		nDCG = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		nDCG_list = []
		for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
			query_id = str(query_id)
			q_nDCG = self.queryNDCG(doc_id_order, query_id, qrels, k)
			nDCG_list.append(q_nDCG)
		meanNDCG = sum(nDCG_list) / len(nDCG_list) if nDCG_list else 0.0
		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		rel_precision_list = []
		for index, doc_id in enumerate(query_doc_IDs_ordered[:k]):
			if doc_id in true_doc_IDs:
				q_precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, index+1)
				rel_precision_list.append(q_precision)
		avgPrecision = sum(rel_precision_list) / len(rel_precision_list) if rel_precision_list else 0.0
		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""
		meanAveragePrecision = -1

		#Fill in code here
		avg_precision_list = []
		for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
			query_id = str(query_id)
			rel_doc_ids = [x["id"] for x in q_rels if x["query_num"] == query_id]
			qavg_precision = self.queryAveragePrecision(doc_id_order, query_id, rel_doc_ids, k)
			avg_precision_list.append(qavg_precision)
		meanAveragePrecision = sum(avg_precision_list) / len(avg_precision_list) if avg_precision_list else 0.0
		return meanAveragePrecision

