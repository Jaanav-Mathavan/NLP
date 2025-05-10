<<<<<<< HEAD
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

class TFIDF:
    def __init__(self, qex=False, dex=False, include_bigrams=False):
        """
        Parameters
        ----------
        qex : bool
            Whether to use query expansion (default False).
        dex : bool
            Whether to use document expansion (default False).
        """
        self.vectorizer = None
        self.doc_tfidf_matrix = None
        self.words = None
        self.include_bigrams = include_bigrams
        self.idf_list = None
        self.qex = qex  
        self.dex = dex  

    def fit(self, documents):
        if self.dex:
            if self.include_bigrams:
                temp_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            else:
                temp_vectorizer = TfidfVectorizer()
            temp_vectorizer.fit(documents)
            vocab = set(temp_vectorizer.get_feature_names_out())
            documents = self.expand_documents(documents, vocab)
        if self.include_bigrams:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        else:
            self.vectorizer = TfidfVectorizer()
        self.doc_tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.words = set(self.vectorizer.get_feature_names_out())
        self.idf_list = self.vectorizer.idf_

    def transform(self, queries):
        if self.qex:
            queries = self.expand_queries(queries)
        return self.vectorizer.transform(queries)

    def get_doc_matrix(self):
        return self.doc_tfidf_matrix
=======
import os
import json
import numpy as np
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

class TFIDF:
    def __init__(
        self,
        qex: bool = False,
        dex: bool = False,
        include_bigrams: bool = False,
        use_wn: bool = False,
        sim_matrix_path: str = "models/wu_palmer_similarity_matrix_2.npy",
        alpha: float = 0.2,  # smoothing strength
        propagate_to_docs: bool = False
    ):
        self.qex = qex
        self.dex = dex
        self.include_bigrams = include_bigrams
        self.use_wn = use_wn
        self.alpha = alpha
        self.propagate_to_docs = propagate_to_docs

        self.vectorizer = None
        self.doc_tfidf_matrix = None
        self.words = None
        self.idf_list = None

        self.sim_matrix_path = sim_matrix_path
        self.sim_matrix = None

    def fit(self, documents: list[str], vocab_path: str = "models/vocab.json"):
        # Optionally expand documents
        if self.dex:
            temp_vectorizer = TfidfVectorizer(ngram_range=(1, 2)) if self.include_bigrams else TfidfVectorizer()
            temp_vectorizer.fit(documents)
            vocab = temp_vectorizer.get_feature_names_out()
            documents = self.expand_documents(documents, vocab)

        # Load fixed vocabulary
        with open(vocab_path, "r") as f:
            vocab_list = json.load(f)
        self.words = vocab_list

        # Create vectorizer with fixed vocab
        self.vectorizer = TfidfVectorizer(vocabulary=vocab_list, ngram_range=(1, 2) if self.include_bigrams else (1, 1))
        self.doc_tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.idf_list = self.vectorizer.idf_

        # Load similarity matrix if needed
        if self.use_wn:
            if not os.path.exists(self.sim_matrix_path):
                raise FileNotFoundError(f"Similarity matrix not found at {self.sim_matrix_path}")
            self.sim_matrix = np.load(self.sim_matrix_path)
            if self.sim_matrix.shape != (len(self.words), len(self.words)):
                raise ValueError("Mismatch between vocab size and similarity matrix shape")

            if self.propagate_to_docs:
                raw_doc = self.doc_tfidf_matrix.toarray()
                prop_doc = raw_doc @ self.sim_matrix
                blended_doc = normalize((1- self.alpha)*raw_doc + self.alpha * prop_doc, norm="l2")
                self.doc_tfidf_matrix = blended_doc

    def transform(self, queries: list[str]) -> np.ndarray:
        # Optionally expand queries
        if self.qex:
            queries = self.expand_queries(queries)

        tfidf_q = self.vectorizer.transform(queries)
        raw_q = tfidf_q.toarray()

        if self.use_wn and self.sim_matrix is not None:
            prop = tfidf_q @ self.sim_matrix
            prop_q = prop.toarray() if hasattr(prop, "toarray") else np.asarray(prop)
            blended = normalize((1-self.alpha)*raw_q + self.alpha * prop_q, norm="l2")
            return blended

        return raw_q

    def get_doc_matrix(self):
        return self.doc_tfidf_matrix.toarray()
>>>>>>> 46b8c50 (.)

    def get_words(self):
        return self.words

    def get_idf(self):
        return self.idf_list

    def expand_queries(self, queries):
        expanded_queries = []
        for query in queries:
            expanded_query = []
            for word in query.split():
<<<<<<< HEAD
                expanded_query.append(word)  
                if wordnet.synsets(word):
                    synonyms = set()
                    for syn in wordnet.synsets(word):
                        for lemma in syn.lemmas():
                            synonym = lemma.name().replace("_", " ")
                            if synonym in self.words:  
                                synonyms.add(synonym)
                    expanded_query.extend(synonyms)
=======
                expanded_query.append(word)
                synonyms = self.get_synonyms(word)
                expanded_query.extend([s for s in synonyms if s in self.words])
>>>>>>> 46b8c50 (.)
            expanded_queries.append(" ".join(expanded_query))
        return expanded_queries

    def expand_documents(self, documents, vocab):
        expanded_documents = []
        for doc in documents:
            expanded_doc = []
            for word in doc.split():
<<<<<<< HEAD
                expanded_doc.append(word)  
                if wordnet.synsets(word):
                    synonyms = set()
                    for syn in wordnet.synsets(word):
                        for lemma in syn.lemmas():
                            synonym = lemma.name().replace("_", " ")
                            if synonym in vocab: 
                                synonyms.add(synonym)
                    expanded_doc.extend(synonyms)
            expanded_documents.append(" ".join(expanded_doc))
        return expanded_documents
=======
                expanded_doc.append(word)
                synonyms = self.get_synonyms(word)
                expanded_doc.extend([s for s in synonyms if s in vocab])
            expanded_documents.append(" ".join(expanded_doc))
        return expanded_documents

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                synonyms.add(synonym)
        return synonyms
>>>>>>> 46b8c50 (.)
