import requests
import time
import pickle
from collections import defaultdict, Counter
from math import log
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class WikipediaIndexBuilder:
    def __init__(self, output_file='wikipedia_index.pkl', max_articles=10000, batch_size=20, max_retries=3):
        """
        Initialize the Wikipedia index builder.

        Parameters:
        - output_file (str): Path to save the pickled index
        - max_articles (int): Maximum number of articles to process
        - batch_size (int): Number of articles per API request
        - max_retries (int): Maximum retries for failed requests
        """
        self.output_file = output_file
        self.max_articles = max_articles
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.session = requests.Session()
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        self.URL = "https://en.wikipedia.org/w/api.php"

    def get_category_members(self, category, depth=0, max_depth=3):
        """
        Recursively get articles in a category and subcategories up to max_depth.

        Parameters:
        - category (str): Wikipedia category name, e.g., "Category:Aerospace engineering"
        - depth (int): Current recursion depth
        - max_depth (int): Maximum recursion depth

        Returns:
        - set: Set of unique article titles
        """
        if depth > max_depth:
            return set()
        articles = set()
        PARAMS = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "page|subcat",
            "cmlimit": "500",
            "format": "json"
        }
        retries = 0
        while retries < self.max_retries:
            try:
                response = self.session.get(url=self.URL, params=PARAMS, timeout=10)
                response.raise_for_status()
                data = response.json()
                members = data["query"]["categorymembers"]
                articles.update(member["title"] for member in members if member["ns"] == 0)
                subcats = [member["title"] for member in members if member["ns"] == 14]
                for subcat in subcats:
                    articles.update(self.get_category_members(subcat, depth + 1, max_depth))
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = 2 ** retries
                    print(f"Rate limit hit for {category}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    print(f"HTTP error for {category}: {e}")
                    break
            except requests.RequestException as e:
                print(f"Request error for {category}: {e}")
                break
            time.sleep(0.5)  # Reduced delay
        return articles

    def get_article_extracts(self, titles):
        """
        Fetch plain text extracts for a list of article titles in batches.

        Parameters:
        - titles (list): List of article titles

        Returns:
        - list: List of tuples (title, extract)
        """
        extracts = []
        titles = list(set(titles))  # Remove duplicates
        for i in range(0, len(titles), self.batch_size):
            batch = titles[i:i + self.batch_size]
            retries = 0
            while retries < self.max_retries:
                try:
                    PARAMS = {
                        "action": "query",
                        "prop": "extracts",
                        "exintro": "",
                        "explaintext": "",
                        "titles": "|".join(batch),
                        "format": "json"
                    }
                    response = self.session.get(url=self.URL, params=PARAMS, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    pages = data["query"]["pages"]
                    for page_id, page in pages.items():
                        if "extract" in page and page["extract"].strip():
                            extracts.append((page["title"], page["extract"]))
                    break
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        wait_time = 2 ** retries
                        print(f"Rate limit hit for batch {i//self.batch_size}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        retries += 1
                    else:
                        print(f"HTTP error for batch {i//self.batch_size}: {e}")
                        break
                except requests.RequestException as e:
                    print(f"Request error for batch {i//self.batch_size}: {e}")
                    break
                time.sleep(0.5)  # Reduced delay
        return extracts

    def preprocess_text(self, text):
        """
        Preprocess text to match Cranfield tokenization (e.g., stemming, no punctuation).

        Parameters:
        - text (str): Raw text

        Returns:
        - list: List of processed tokens
        """
        # Tokenize, remove punctuation, lowercase
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stopwords]
        return tokens

    def build_wikipedia_index_from_extracts(self, extracts):
        """
        Build a term-to-concept inverted index from Wikipedia article extracts.

        Parameters:
        - extracts (list): List of tuples (title, text)

        Returns:
        - dict: Index data to be pickled
        """
        index = defaultdict(list)
        concept_id_to_title = {}
        term_df = defaultdict(int)
        doc_count = min(len(extracts), self.max_articles)

        for concept_id, (title, content) in enumerate(extracts[:self.max_articles]):
            if not content.strip():
                continue
            concept_id_to_title[concept_id] = title
            tokens = self.preprocess_text(content)
            if not tokens:
                continue
            term_freq = Counter(tokens)
            for term in term_freq:
                term_df[term] += 1
            for term, freq in term_freq.items():
                tf = 1 + log(freq)
                index[term].append((concept_id, tf))

        # Compute IDF
        idf = {term: log(doc_count / df) if df > 0 else 0.0 for term, df in term_df.items()}

        # Update index with TF-IDF
        for term in index:
            for i, (concept_id, tf) in enumerate(index[term]):
                tf_idf = tf * idf[term]
                index[term][i] = (concept_id, tf_idf)

        # Verify key terms
        key_terms = ['aeroelast', 'aerodynam', 'slipstream', 'lift']
        for term in key_terms:
            print(f"Term '{term}' in index: {term in index}, TF-IDF entries: {len(index[term]) if term in index else 0}")

        return {
            'index': index,
            'concept_id_to_title': concept_id_to_title,
            'idf': idf,
            'doc_count': doc_count
        }

    def save_index(self, index_data):
        """
        Save the index to a file.

        Parameters:
        - index_data (dict): Index data to pickle
        """
        with open(self.output_file, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"Built and saved index with {index_data['doc_count']} articles and {len(index_data['index'])} unique terms to {self.output_file}")

    def main(self):
        """
        Main function to build the Wikipedia index.
        """
        print("Starting Wikipedia index building...")
        # Broader aerospace-related categories
        categories = [
            "Category:Aerospace engineering",
            "Category:Aeronautics",
            "Category:Aircraft",
            "Category:Fluid dynamics",
            "Category:Mechanical engineering"
        ]
        articles = set()
        for category in categories:
            print(f"Fetching articles from {category}...")
            articles.update(self.get_category_members(category))
        articles = list(articles)
        print(f"Found {len(articles)} unique articles")

        if articles:
            print(f"Fetching extracts for {min(len(articles), self.max_articles)} articles...")
            extracts = self.get_article_extracts(articles)
            print(f"Fetched {len(extracts)} extracts")

            if extracts:
                index_data = self.build_wikipedia_index_from_extracts(extracts)
                self.save_index(index_data)
            else:
                print("No extracts fetched, check API connectivity")
        else:
            print("No articles found, check categories or API connectivity")

if __name__ == "__main__":
    builder = WikipediaIndexBuilder(output_file='wikipedia_index.pkl', max_articles=10000)
    builder.main()
