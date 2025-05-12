import os
import json
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import numpy as np

# Ensure WordNet is downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')  # Sometimes needed for full WordNet support

# Load vocab
vocab_path = "models/vocab.json"
assert os.path.exists(vocab_path), f"Vocab file not found at {vocab_path}"
with open(vocab_path, 'r') as f:
    vocab = json.load(f)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Clean and lemmatize a word
def get_valid_wn_word(word):
    lemma = lemmatizer.lemmatize(word)
    if wn.synsets(lemma):
        return lemma
    return None

# Wu-Palmer similarity calculation with lemmatization
def wu_palmer_similarity(word1, word2):
    w1 = get_valid_wn_word(word1)
    w2 = get_valid_wn_word(word2)
    if not w1 or not w2:
        return 0.0
    syns1 = wn.synsets(w1)
    syns2 = wn.synsets(w2)
    if not syns1 or not syns2:
        return 0.0
    return max((syn1.wup_similarity(syn2) or 0.0) for syn1 in syns1 for syn2 in syns2)

# Compute one row of the matrix
def compute_row(i, vocab):
    return [wu_palmer_similarity(vocab[i], vocab[j]) for j in range(len(vocab))]

# Parallel matrix computation
def compute_similarity_matrix(vocab):
    with Pool(cpu_count()) as pool:
        result = list(tqdm(pool.imap(partial(compute_row, vocab=vocab), range(len(vocab))), total=len(vocab)))
    return np.array(result)

if __name__ == '__main__':
    similarity_matrix = compute_similarity_matrix(vocab)
    output_path = "models/wu_palmer_similarity_matrix.npy"
    np.save(output_path, similarity_matrix)
    print(f"[SUCCESS] Wu-Palmer similarity matrix saved at '{output_path}'")
