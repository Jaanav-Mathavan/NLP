# Add your import statements here
import json
from collections import Counter

# Add any utility functions here
def get_corpus_specific_stopwords(method):
    # Load reducedDocs which was saved in reduced_docs.txt
    reducedDocs = json.load(open(f"output_{method}/reduced_docs.txt", 'r'))
    all_words = [word for doc in reducedDocs for segment in doc for word in segment]
    # Count word frequencies
    term_counts = Counter(all_words)
    # Calculate term frequency (count of word / total words)
    term_frequency = {word: count / len(all_words) for word, count in term_counts.items()}
    # Frequency threshold
    threshold = 0.0013
    custom_stopwords = {word for word, freq in term_frequency.items() if freq > threshold}
    #Sorting the custom stopwords based on their frequency of occurence
    sorted_stopwords = [word for word, freq in sorted(term_frequency.items(), key=lambda x: x[1], reverse=True) if word in custom_stopwords]
    print(f"Identified {len(custom_stopwords)} stopwords based on frequency.")
    #output saved to the below output path in a folder output_punkt if you chose the bottom up approach and output_naive if you chose the naive top down approach
    output_path = f"output_{method}/custom_stopwords_using_frequency.txt"
    with open(output_path, 'w') as f:
        f.write(str(sorted_stopwords))

    print(f"Stopwords saved to: {output_path}")
