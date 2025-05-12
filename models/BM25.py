import numpy as np
from collections import Counter, defaultdict
from util import *

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # term frequency saturation parameter
        self.b = b    # length normalization parameter
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.N = 0
        self.index = None
        self.doc_freqs = None
        
    def fit(self, docs, doc_ids):
        """Build the BM25 model from documents"""
        self.N = len(docs)
        
        # Build inverted index and document lengths
        index = defaultdict(list)
        doc_freqs = defaultdict(dict)
        total_length = 0
        
        for doc, doc_id in zip(docs, doc_ids):
            words = [word for sentence in doc for word in sentence]
            
            # Store document length
            doc_length = len(words)
            self.doc_lengths[doc_id] = doc_length
            total_length += doc_length
            
            # Count term frequencies
            term_freqs = Counter(words)
            
            # Update inverted index and document frequencies
            for word, freq in term_freqs.items():
                index[word].append(doc_id)
                doc_freqs[word][doc_id] = freq

        self.avg_doc_length = total_length / self.N
        self.index = {word: sorted(doc_ids) for word, doc_ids in index.items()}
        self.doc_freqs = doc_freqs
        
    def score(self, query_terms, doc_id):
        """Calculate BM25 score for a document"""
        score = 0.0
        doc_length = self.doc_lengths[doc_id]
        
        for term in query_terms:
            if term not in self.index:
                continue
                
            # Calculate IDF component
            n_docs_with_term = len(self.index[term])
            idf = np.log((self.N - n_docs_with_term + 0.5) / (n_docs_with_term + 0.5) + 1)
            
            # Calculate TF component
            tf = self.doc_freqs[term].get(doc_id, 0)
            tf_component = ((tf * (self.k1 + 1)) / 
                          (tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)))
            
            score += idf * tf_component
            
        return score

    def get_relevant_docs(self, query_terms):
        """Get all documents that contain at least one query term"""
        relevant_docs = set()
        for term in query_terms:
            if term in self.index:
                relevant_docs.update(self.index[term])
        return relevant_docs

class IR_BM25():
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