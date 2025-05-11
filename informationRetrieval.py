from util import *
import numpy as np
from models.BM25 import BM25

class InformationRetrieval():
    def __init__(self):
        self.bm25 = BM25()
        
    def buildIndex(self, docs, docIDs):
        """
        Builds the document index and fits BM25 model
        """
        self.bm25.fit(docs, docIDs)

    def rank(self, queries):
        """
        Rank documents using BM25 scoring
        """
        doc_IDs_ordered = []

        for query in queries:
            # Flatten query into terms
            query_terms = [word for sentence in query for word in sentence]
            
            # Get relevant documents
            relevant_docs = self.bm25.get_relevant_docs(query_terms)

            # Score documents
            doc_scores = []
            for doc_id in relevant_docs:
                score = self.bm25.score(query_terms, doc_id)
                doc_scores.append((str(doc_id), score))
            
            # Sort by score in descending order
            ranked_docs = [doc_id for doc_id, _ in sorted(doc_scores, key=lambda x: x[1], reverse=True)]
            doc_IDs_ordered.append(ranked_docs)

        return doc_IDs_ordered