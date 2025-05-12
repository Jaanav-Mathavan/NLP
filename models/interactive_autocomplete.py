from collections import defaultdict, Counter
import difflib
import json

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

    def train(self, queries):
        self.vocabulary = set(word for q in queries for word in q.split())
        self.queries = queries
        for query in queries:
            self.insert(query)

    def _dfs(self, node, prefix, suggestions):
        if node.is_end_of_word:
            suggestions.append((prefix, node.frequency))
        for char, next_node in node.children.items():
            self._dfs(next_node, prefix + char, suggestions)

    def autocomplete(self, prefix, top_k=5):
        node = self.root
        for char in prefix:
            if char not in node.children:
                corrected_prefix = self.spell_correct(prefix)
                if corrected_prefix == prefix:
                    return [prefix]
                else:
                    return self.autocomplete(corrected_prefix, top_k=top_k)
            node = node.children[char]

        suggestions = []
        self._dfs(node, prefix, suggestions)
        suggestions = sorted(suggestions, key=lambda x: -x[1])
        return [s[0] for s in suggestions[:top_k]] if suggestions else [prefix]

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
        self.model = Trie()

    def complete(self, query, top_k=5):
        return self.model.autocomplete(query, top_k=top_k)


def interactive_autocomplete(model, query, top_k=5):
    suggestions = model.complete(query, top_k=top_k)
    print("\nSuggestions:")
    for idx, s in enumerate(suggestions):
        print(f"{idx + 1}. {s}")
    choice = input("Choose one (1-{}), or press Enter to select top suggestion: ".format(len(suggestions)))
    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(suggestions):
            return suggestions[index]
    return suggestions[0]  # default fallback

if __name__ == "__main__":
    queries_json = json.load(open("cran_queries.json", 'r'))
    queries = [item["query"] for item in queries_json]
    
    print("\n--- Trie-based Autocomplete ---")
    model_trie = Autocomplete(model='tries')
    model_trie.model.train(queries)
    result = interactive_autocomplete(model_trie, input("Enter your Trie query: "))
    print("Final Trie Suggestion:", result)
    