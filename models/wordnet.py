from nltk.corpus import wordnet as wn
import numpy as np

class WordNetSimilarity:
    def __init__(self, vocab):
        self.vocab = vocab
        self.similarity_matrix = None

    def _get_wordnet_similarity(self, word1, word2):
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        if not synsets1 or not synsets2:
            return 0.0
        max_sim = 0
        for s1 in synsets1:
            for s2 in synsets2:
                sim = s1.wup_similarity(s2) # This similarity measure is similar to D to LCA but its 2 D(LCA) / (D(s1) + D(s2))
                if sim is not None:
                    max_sim = max(max_sim, sim)
        return max_sim
    
    def build_similarity_matrix(self):
        vocab_size = len(self.vocab)
        self.similarity_matrix = np.zeros((vocab_size, vocab_size))
        
        for i, word1 in enumerate(self.vocab):
            for j, word2 in enumerate(self.vocab):
                if i != j:
                    sim = self._get_wordnet_similarity(word1, word2)
                    self.similarity_matrix[i][j] = sim
                else:
                    self.similarity_matrix[i][j] = 1.0 
        return self.similarity_matrix

    def transform(self, query_vector):
        transformed_query = np.dot(self.similarity_matrix, query_vector.T)
        return transformed_query
