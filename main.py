from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
# from informationRetrieval import InformationRetrieval
from models.BM25 import IR_BM25
from esa import ExplicitSemanticAnalysis
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
from models.autocomplete import Autocomplete
import os
from sys import version_info
import argparse
import time
import json
import matplotlib.pyplot as plt

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print("Unknown python version - input function not safe")

class SearchEngine:
    def __init__(self, args):
        self.args = args
        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()
        self.evaluator = Evaluation()
        self.autocomplete = Autocomplete(model=self.args.autocomplete,n=self.args.Ngram)
        if args.model in ["esa", "nesa"]:
            self.informationRetriever = ExplicitSemanticAnalysis(model_type=args.model)
        elif args.model == "bm25":
            self.informationRetriever = IR_BM25()
        else:
            self.informationRetriever = InformationRetrieval()

    def segmentSentences(self, text):
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)
    def segmentSentences(self, text):
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)
    def tokenize(self, text):
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        return self.inflectionReducer.reduce(text)
    def reduceInflection(self, text):
        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text):
        return self.stopwordRemover.fromList(text)
    def removeStopwords(self, text):
        return self.stopwordRemover.fromList(text)

    def preprocessQueries(self, queries):
        os.makedirs(self.args.out_folder, exist_ok=True)
        self.autocomplete.model.train(queries)
        segmentedQueries = [self.segmentSentences(query) for query in queries]
        json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
        tokenizedQueries = [self.tokenize(query) for query in segmentedQueries]
        json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
        reducedQueries = [self.reduceInflection(query) for query in tokenizedQueries]
        json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
        stopwordRemovedQueries = [self.removeStopwords(query) for query in reducedQueries]
        json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))
        print("Sample preprocessed query tokens:", stopwordRemovedQueries[0][:10])
        return stopwordRemovedQueries
    def preprocessQueries(self, queries):
        os.makedirs(self.args.out_folder, exist_ok=True)
        self.autocomplete.model.train(queries)
        segmentedQueries = [self.segmentSentences(query) for query in queries]
        json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
        tokenizedQueries = [self.tokenize(query) for query in segmentedQueries]
        json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
        reducedQueries = [self.reduceInflection(query) for query in tokenizedQueries]
        json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
        stopwordRemovedQueries = [self.removeStopwords(query) for query in reducedQueries]
        json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))
        print("Sample preprocessed query tokens:", stopwordRemovedQueries[0][:10])
        return stopwordRemovedQueries

    def preprocessDocs(self, docs):
        segmentedDocs = [self.segmentSentences(doc) for doc in docs]
        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
        tokenizedDocs = [self.tokenize(doc) for doc in segmentedDocs]
        json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
        reducedDocs = [self.reduceInflection(doc) for doc in tokenizedDocs]
        json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
        stopwordRemovedDocs = [self.removeStopwords(doc) for doc in reducedDocs]
        json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))
        print("Sample preprocessed doc tokens:", stopwordRemovedDocs[0][:10])
        return stopwordRemovedDocs
    def preprocessDocs(self, docs):
        segmentedDocs = [self.segmentSentences(doc) for doc in docs]
        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
        tokenizedDocs = [self.tokenize(doc) for doc in segmentedDocs]
        json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
        reducedDocs = [self.reduceInflection(doc) for doc in tokenizedDocs]
        json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
        stopwordRemovedDocs = [self.removeStopwords(doc) for doc in reducedDocs]
        json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))
        print("Sample preprocessed doc tokens:", stopwordRemovedDocs[0][:10])
        return stopwordRemovedDocs

    def evaluateDataset(self):
        queries_json = json.load(open(self.args.dataset + "cran_queries.json", 'r'))[:]
        query_ids = [str(item["query number"]) for item in queries_json]  # Ensure string IDs
        queries = [item["query"] for item in queries_json]
        processedQueries = self.preprocessQueries(queries)

        docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids = [str(item["id"]) for item in docs_json]  # Ensure string IDs
        docs = [item["body"] for item in docs_json]
        processedDocs = self.preprocessDocs(docs)

        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        doc_IDs_ordered = self.informationRetriever.rank(processedQueries)

        qrels = json.load(open(self.args.dataset + "cran_qrels.json", 'r'))[:]
        rank=20
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        for k in range(1, rank+1):
            precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k)
            precisions.append(precision)
            recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)
            fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)
            print("Precision, Recall and F-score @ " +  
				str(k) + " : " + str(precision) + ", " + str(recall) + 
				", " + str(fscore))
            MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k)
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " +  
				str(k) + " : " + str(MAP) + ", " + str(nDCG))

        # Plot the metrics and save plot 
        plt.plot(range(1, rank+1), precisions, label="Precision")
        plt.plot(range(1, rank+1), recalls, label="Recall")
        plt.plot(range(1, rank+1), fscores, label="F-Score")
        plt.plot(range(1, rank+1), MAPs, label="MAP")
        plt.plot(range(1, rank+1), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(args.out_folder + "eval_plot.png")

    def handleCustomQuery(self):
        """
        Take a custom query as input and return top five relevant documents
        """

        # Get query
        print("Enter query below")
        query = input()
        custom_start_time = time.time()
        # Process documents
        query = self.autocomplete.complete(query)
        processedQuery = self.preprocessQueries([query])[0]
        # Get query
        print("Enter query below")
        query = input()
        custom_start_time = time.time()
        # Process documents
        query = self.autocomplete.complete(query)
        processedQuery = self.preprocessQueries([query])[0]

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
                        [item["body"] for item in docs_json]
        # Process documents
        processedDocs = self.preprocessDocs(docs)
        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
                        [item["body"] for item in docs_json]
        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # Rank the documents for the query
        doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]
        custom_end_time = time.time()
        print("Custom Query Time taken : " + str(custom_end_time - custom_start_time) + " seconds")
        # Print the IDs of first five documents
        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)
        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # Rank the documents for the query
        doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]
        custom_end_time = time.time()
        print("Custom Query Time taken : " + str(custom_end_time - custom_start_time) + " seconds")
        # Print the IDs of first five documents
        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)

if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-model', default= "tfidf", choices=['tfidf', 'lsa', 'hybrid', 'esa', 'nesa', 'bm25', 'wordnet_tfidf', 'wordnet_lsa', 'wordnet_esa', 'wordnet_hybrid', 'embeddings'], 
                    help="Choose the model: 'tfidf', 'lsa', 'esa', or 'hybrid'")
	parser.add_argument('-qex', default=False, action = "store_true",
						help = "Use query expansion")
	parser.add_argument('-dex', default=False, action = "store_true",
						help = "Use document expansion")
	parser.add_argument('-model', default= "tfidf", choices=['tfidf', 'lsa', 'hybrid', 'esa', 'nesa', 'bm25', 'wordnet_tfidf', 'wordnet_lsa', 'wordnet_esa', 'wordnet_hybrid', 'embeddings'], 
                    help="Choose the model: 'tfidf', 'lsa', 'esa', or 'hybrid'")
	parser.add_argument('-qex', default=False, action = "store_true",
						help = "Use query expansion")
	parser.add_argument('-dex', default=False, action = "store_true",
						help = "Use document expansion")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	parser.add_argument('-include_bigrams', action="store_true",
						help="Include bigrams in the TF-IDF model")
	parser.add_argument('-autocomplete',default= "Ngram", choices=['Ngram', 'tries'], 
						help="Autocomplete Queries")
	parser.add_argument('-Ngram',default=3, choices=[2,3,4,5], 
						help="Ngram Model type")
	
	parser.add_argument('-include_bigrams', action="store_true",
						help="Include bigrams in the TF-IDF model")
	parser.add_argument('-autocomplete',default= "Ngram", choices=['Ngram', 'tries'], 
						help="Autocomplete Queries")
	
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	start_time = time.time()
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()
		end_time = time.time()
		print("Cranfield Query Dataset Time taken : " + str(end_time - start_time) + " seconds")
	
	