from util import *
import numpy as np
import os
import pandas as pd
# Add your import statements here
from pandas.errors import EmptyDataError

class Evaluation:
    def __init__(self, model):
        self.model = model

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
    
        # Fill in code here
        self.precision_list = []
        try:
            self.meanPrecisiondata = pd.read_csv('ttest_tables/meanPrecision.csv')
        except EmptyDataError:
            self.meanPrecisiondata = pd.DataFrame()
        for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
            query_id = str(query_id)
            rel_doc_ids = [x["id"] for x in qrels if x["query_num"] == query_id]
            q_precision = self.queryPrecision(doc_id_order, query_id, rel_doc_ids, k)
            self.precision_list.append(q_precision)
        self.meanPrecisiondata[self.model] = self.precision_list
        self.meanPrecisiondata.to_csv('ttest_tables/meanPrecision.csv', index=False)
        meanPrecision = sum(self.precision_list) / len(self.precision_list) if self.precision_list else 0.0
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

        # Fill in code here
        self.recall_list = []
        try:
            self.meanRecalldata = pd.read_csv('ttest_tables/meanRecall.csv')
        except EmptyDataError:
            self.meanRecalldata = pd.DataFrame()
        for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
            query_id = str(query_id)
            rel_doc_ids = [x["id"] for x in qrels if x["query_num"] == query_id]
            q_recall = self.queryRecall(doc_id_order, query_id, rel_doc_ids, k)
            self.recall_list.append(q_recall)
        self.meanRecalldata[self.model] = self.recall_list
        self.meanRecalldata.to_csv('ttest_tables/meanRecall.csv', index=False)
        meanRecall = sum(self.recall_list) / len(self.recall_list) if self.recall_list else 0.0
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
        Computation of F1-score of the Information Retrieval System
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
            The mean F1-score value as a number between 0 and 1
        """

        meanFscore = -1

        # Fill in code here
        self.fscore_list = []
        try:
            self.meanFscoredata = pd.read_csv('ttest_tables/meanFscore.csv')
        except EmptyDataError:
            self.meanFscoredata = pd.DataFrame()
        for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
            query_id = str(query_id)
            rel_doc_ids = [x["id"] for x in qrels if x["query_num"] == query_id]
            q_fscore = self.queryFscore(doc_id_order, query_id, rel_doc_ids, k)
            self.fscore_list.append(q_fscore)
        self.meanFscoredata[self.model] = self.fscore_list
        self.meanFscoredata.to_csv('ttest_tables/meanFscore.csv', index=False)
        meanFscore = sum(self.fscore_list) / len(self.fscore_list) if self.fscore_list else 0.0
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
        Computation of Normalized Discounted Cumulative Gain (nDCG) of the Information Retrieval System
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

        # Fill in code here
        self.ndcg_list = []
        try:
            self.meanNDCGdata = pd.read_csv('ttest_tables/meanNDCG.csv')
        except EmptyDataError:
            self.meanNDCGdata = pd.DataFrame()
        for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
            query_id = str(query_id)
            q_ndcg = self.queryNDCG(doc_id_order, query_id, qrels, k)
            self.ndcg_list.append(q_ndcg)
        self.meanNDCGdata[self.model] = self.ndcg_list
        self.meanNDCGdata.to_csv('ttest_tables/meanNDCG.csv', index=False)
        meanNDCG = sum(self.ndcg_list) / len(self.ndcg_list) if self.ndcg_list else 0.0
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


    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of average precision of the Information Retrieval System
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
            The mean average precision value as a number between 0 and 1
        """

        meanAveragePrecision = -1

        # Fill in code here
        self.ap_list = []
        try:
            self.meanAPdata = pd.read_csv('ttest_tables/meanAveragePrecision.csv')
        except EmptyDataError:
            self.meanAPdata = pd.DataFrame()
        for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
            query_id = str(query_id)
            rel_doc_ids = [x["id"] for x in qrels if x["query_num"] == query_id]
            q_ap = self.queryAveragePrecision(doc_id_order, query_id, rel_doc_ids, k)
            self.ap_list.append(q_ap)
        self.meanAPdata[self.model] = self.ap_list
        self.meanAPdata.to_csv('ttest_tables/meanAveragePrecision.csv', index=False)
        meanAveragePrecision = sum(self.ap_list) / len(self.ap_list) if self.ap_list else 0.0
        return meanAveragePrecision
