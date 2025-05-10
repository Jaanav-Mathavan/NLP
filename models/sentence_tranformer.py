import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import nltk
from nltk.tokenize import sent_tokenize

class SentenceTransformerEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Parameters
        ----------
        model_name : str
            Pre-trained model name from HuggingFace.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.doc_matrix = None

    def fit(self, doc_term_matrix):
        """
        Prepare document embeddings by processing each document.
        """
        self.documents = doc_term_matrix
        self.doc_matrix = np.vstack([self.embed_long_document(doc) for doc in doc_term_matrix])

    def transform(self, queries_term_matrix):
        """
        Generate embeddings for queries.
        """
        query_matrix = np.vstack([self.embed_long_document(query) for query in queries_term_matrix])
        return query_matrix

    def embed_long_document(self, document):
        """
        Embed a long document by splitting into sentences and averaging embeddings.
        """
        sentences = sent_tokenize(document)
        sentence_embeddings = [self.embed_sentence(sent) for sent in sentences if sent.strip()]
        
        if not sentence_embeddings:
            return np.zeros(self.model.config.hidden_size)
        
        return np.mean(sentence_embeddings, axis=0)

    def embed_sentence(self, sentence):
        """
        Embed a single sentence into a vector using the Sentence Transformer model.
        """
        inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings

    def get_doc_matrix(self):
        return self.doc_matrix
