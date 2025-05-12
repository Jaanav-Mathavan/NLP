
from collections import defaultdict, Counter
import difflib
import os
import json
import heapq

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0 

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, freq=1):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.frequency += freq
    
    def train(self,queries):
        self.vocabulary = set(word for q in queries for word in q.split())
        self.queries = queries
        for query in queries:
            self.insert(query)

    def _dfs(self, node, prefix, suggestions):
        if node.is_end_of_word:
            suggestions.append((prefix, node.frequency))
        for char, next_node in node.children.items():
            self._dfs(next_node, prefix + char, suggestions)

    def autocomplete(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                # No direct match, try spell correction
                corrected_prefix = self.spell_correct(prefix)
                if corrected_prefix==prefix:
                    return prefix
                else:
                    return self.autocomplete(corrected_prefix)
            node = node.children[char]
        
        suggestions = []
        self._dfs(node, prefix, suggestions)
        
        autocompleted_query = sorted(suggestions, key=lambda x: -x[1])
        return autocompleted_query[0][0] if autocompleted_query else prefix
    
    def spell_correct(self, query):
        tokens = query.split()
        if not tokens:
            return query
        corrected_tokens = []
        for token in tokens:
            closest = difflib.get_close_matches(token,self.vocabulary, n=1)
            if closest:
                corrected_tokens.append(closest[0])
            else:
                corrected_tokens.append(token)
        return ' '.join(corrected_tokens)

class NgramAutocomplete:
    def __init__(self,n=3):
        self.n = n
        self.ngrams = defaultdict(Counter)
        

    def train(self, queries):
        self.vocabulary = set(word for q in queries for word in q.split())
        self.queries = queries
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
            if count>10:
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
                corrected_query = self.spell_correct(query, self.vocabulary) #No such query found correct the spelling
                if corrected_query == query:
                    break
                else:
                    query = corrected_query
                    continue  
            
        return query

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
    def __init__(self, model='tries', n=3):
        self.model_type = model
        if model == 'tries':
            self.model = Trie()
        elif model == 'Ngram':
            self.model = NgramAutocomplete(n)
        else:
            raise ValueError("Invalid model. Choose 'tries' or 'Ngram'")
        

    def train(self, queries):
        self.model.train(self.queries)

    def complete(self, query):
        if self.model_type == 'tries':
            return self.model.autocomplete(query)
        elif self.model_type == 'Ngram':
            return self.model.autocomplete(query)

if __name__ == "__main__":
    queries_json = json.load(open("cran_queries.json", 'r'))[:]
    query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]

    # Using Trie model
    model_trie = Autocomplete(model='tries')
    model_trie.model.train(queries)
    print("Trie Autocomplete:", model_trie.complete("des a paccal fw"))

    # Using N-gram model
    model_ngram = Autocomplete(model='Ngram', n=3)
    model_ngram.model.train(queries)
    print("N-gram Autocomplete:", model_ngram.complete("what chemical kinetic system"))
