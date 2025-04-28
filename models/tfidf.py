from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDF:
    def __init__(self):
        self.vectorizer = None
        self.doc_tfidf_matrix = None
        self.words = None
        self.idf_list = None

    def fit(self, documents):
        self.vectorizer = TfidfVectorizer()
        self.doc_tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.words = self.vectorizer.get_feature_names_out()
        self.idf_list = self.vectorizer.idf_

    def transform(self, queries):
        return self.vectorizer.transform(queries)

    def get_doc_matrix(self):
        return self.doc_tfidf_matrix

    def get_words(self):
        return self.words

    def get_idf(self):
        return self.idf_list
