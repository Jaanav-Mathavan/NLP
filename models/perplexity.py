import random
import math
from collections import defaultdict,Counter
import difflib
import matplotlib.pyplot as plt
import json
import math

class NgramAutocomplete:
    def __init__(self,n=3):
        self.n = n
        self.ngrams = defaultdict(Counter)

    def train(self, queries):
        self.queries = queries
        self.vocabulary = set(word for q in queries for word in q.split())
        for sentence in queries:
            tokens = sentence.lower().split()
            for i in range(len(tokens) - self.n + 1):
                prefix = tuple(tokens[i:i+self.n-1])
                next_word = tokens[i+self.n-1]
                self.ngrams[prefix][next_word] += 1

    def predict(self, prefix):
        tokens = prefix.lower().split()
        complete=True
        if not tokens:
            return False

        if len(tokens) >= self.n - 1:
            prefix_tuple = tuple(tokens[-(self.n - 1):])
        else:
            prefix_tuple = tuple(tokens)

        suggestions = self.ngrams.get(prefix_tuple, {})

        if suggestions:
            return suggestions.most_common(),complete

        # No direct match, assume last word is incomplete
        complete = False
        if len(tokens) > self.n:
            prefix_tuple = tuple(tokens[-(self.n):-1])
        else:
            prefix_tuple = tuple(tokens[:-1])
        last_incomplete = tokens[-1]
        suggestions = self.ngrams.get(prefix_tuple, {})
        if not suggestions:
            return False,complete

        filtered = [(word, count) for word, count in suggestions.items() if word.startswith(last_incomplete)]
        if not filtered:
            return False,complete

        return sorted(filtered, key=lambda x: -x[1]),complete
    
    def autocomplete(self,query):
        count=0
        while query not in self.queries:
            predictions,complete = self.predict(query)
            count+=1
            if count>30:
                break
            if predictions:
                
                next_word = predictions[0][0]
                if complete:
                    query += " " + next_word
                else:
                    tokens = query.split()
                    tokens[-1] = next_word  # replace last incomplete word
                    query = ' '.join(tokens)
            else:
                corrected_query = self.spell_correct(query) #No such query found correct the spelling
                if corrected_query == query:
                    break
                else:
                    query = corrected_query
                    continue  
            
        return query

    def perplexity(self, query, autocompleted_query):
        query_tokens = query.lower().split()
        full_tokens = autocompleted_query.lower().split()
        
        # Find where the new tokens start
        start_index = len(query_tokens)
        tokens = full_tokens[start_index:]

        # Not enough tokens to apply n-gram model
        if len(tokens) < self.n - 1:
            return float('inf')
        
        # Reconstruct the full context starting from the last (n-1) tokens of the query
        context = query_tokens[-(self.n - 1):] if len(query_tokens) >= self.n - 1 else query_tokens
        tokens = context + tokens  # full evaluation span

        log_prob_sum = 0
        N = len(tokens) - self.n + 1
        if N <= 0:
            return float('inf')

        for i in range(N):
            prefix = tuple(tokens[i:i + self.n - 1])
            next_word = tokens[i + self.n - 1]
            prefix_counts = self.ngrams.get(prefix, {})
            total = sum(prefix_counts.values())
            count = prefix_counts.get(next_word, 0)

            # Laplace smoothing
            prob = (count + 1) / (total + len(self.vocabulary))
            log_prob_sum += math.log(prob)

        return math.exp(-log_prob_sum / N)



    def spell_correct(self, query):
        tokens = query.split()
        if not tokens:
            return query
        corrected_tokens = []
        for token in tokens:
            closest = difflib.get_close_matches(token, self.vocabulary, n=1)
            if closest:
                corrected_tokens.append(closest[0])
            else:
                corrected_tokens.append(token)
        return ' '.join(corrected_tokens)

class Autocomplete:
    def __init__(self,model='Ngram', n=3):
        self.model_type = model
        self.queries = queries
        self.model = NgramAutocomplete(n)

    def complete(self, query):
        if self.model_type == 'tries':
            return self.model.autocomplete(query)
        elif self.model_type == 'Ngram':
            return self.model.autocomplete(query)


if __name__ == "__main__":
    queries_json = json.load(open("cran_queries.json", 'r'))
    queries = [item["query"] for item in queries_json]
    model_ngram = Autocomplete(model='Ngram', n=3)
    model_ngram.model.train(queries)
    user_query = input("Enter your query: ")
    corrected_query = model_ngram.model.spell_correct(user_query)
    autocompleted_query = model_ngram.complete(user_query)
    perplexity = model_ngram.model.perplexity(corrected_query,autocompleted_query)
    print("N-gram Autocomplete: ", autocompleted_query)
    print("Perplexity: ",perplexity )