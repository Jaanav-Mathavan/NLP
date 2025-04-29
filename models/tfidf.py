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

    def get_words(self):
        return self.words

    def get_idf(self):
        return self.idf_list

    def expand_queries(self, queries):
        expanded_queries = []
        for query in queries:
            expanded_query = []
            for word in query.split():
                expanded_query.append(word)  
                if wordnet.synsets(word):
                    synonyms = set()
                    for syn in wordnet.synsets(word):
                        for lemma in syn.lemmas():
                            synonym = lemma.name().replace("_", " ")
                            if synonym in self.words:  
                                synonyms.add(synonym)
                    expanded_query.extend(synonyms)
            expanded_queries.append(" ".join(expanded_query))
        return expanded_queries

    def expand_documents(self, documents, vocab):
        expanded_documents = []
        for doc in documents:
            expanded_doc = []
            for word in doc.split():
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
