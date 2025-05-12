from sklearn.decomposition import TruncatedSVD

class LSA:
    def __init__(self, n_components=150):
        self.model = None
        self.doc_lsa_matrix = None
        self.n_components = n_components

    def fit(self, doc_term_matrix):
        self.model = TruncatedSVD(n_components=self.n_components)
        self.doc_lsa_matrix = self.model.fit_transform(doc_term_matrix)

    def transform(self, queries_term_matrix):
        return self.model.transform(queries_term_matrix)

    def get_doc_matrix(self):
        return self.doc_lsa_matrix