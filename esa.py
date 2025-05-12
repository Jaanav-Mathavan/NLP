from __future__ import print_function, division
from collections import defaultdict, Counter
from math import log, sqrt
import pickle
import time
# Assuming informationRetrieval.py is in the same directory or accessible in PYTHONPATH
from informationRetrieval import InformationRetrieval
import os # For checking file existence

class ExplicitSemanticAnalysis:
    def __init__(self, model_type="esa", wikipedia_index_file='wikipedia_index.pkl', precompute_path='precomputed_esa_data.pkl'):
        """
        Initialize the Explicit Semantic Analysis system with Wikipedia concept space.

        Parameters
        ----------
        model_type : str
            'esa' for standard ESA, 'nesa' for Normalized ESA.
        wikipedia_index_file : str
            Path to the pre-built Wikipedia term-to-concept index.
        precompute_path : str
            Path to store/load precomputed concept norms and ICF.
        """
        if model_type.lower() not in ["esa", "nesa"]:
            raise ValueError("model_type must be 'esa' or 'nesa'")
        self.model_type = model_type.lower()
        self.wikipedia_index = None
        self.idf = None # IDF for terms from the Wikipedia collection
        self.concept_ids = None
        self.concept_norms = None
        self.icf = None # Inverse Concept Frequency
        self.doc_vectors = {} # Stores {doc_id: (vector, norm)}
        self.doc_count_wikipedia = 0 # Number of concepts/documents in Wikipedia
        self.concept_id_to_title = {}
        self.doc_term_freq = {} # Stores term frequencies for input documents {doc_id: Counter}
        
        # TF-IDF retriever for hybrid scoring and potentially for document term IDF
        self.tfidf_retriever = InformationRetrieval() 

        self.load_wikipedia_index(wikipedia_index_file, precompute_path)

    def _compute_and_save_nesa_params(self, precompute_path):
        """
        Computes concept_norms and icf for NESA and saves them.
        This is called if precomputed data is not found.
        """
        print("Computing NESA parameters (concept_norms and icf)...")
        start_time = time.time()
        
        concept_norms_sq = defaultdict(float) # Store sum of squares of tf-idf scores for each concept
        concept_freq = defaultdict(int) # How many terms map to a concept (used for ICF)

        # Iterate through all terms in the Wikipedia index
        for term in self.wikipedia_index:
            # For each term, iterate through its associated (concept_id, tf_idf_score) pairs
            for concept_id, tf_idf_score in self.wikipedia_index[term]:
                concept_norms_sq[concept_id] += tf_idf_score ** 2
                # Increment frequency for the concept if this term maps to it.
                # Note: This counts term-concept links, not unique documents containing the concept directly.
                # For a more standard ICF, concept_freq should ideally be the number of documents a concept appears in.
                # However, based on the original code, it seems to be derived from term-concept links.
                concept_freq[concept_id] += 1 

        # Calculate L2 norm for each concept vector
        self.concept_norms = {
            cid: sqrt(norm_sq) if norm_sq > 0 else 1.0 
            for cid, norm_sq in concept_norms_sq.items()
        }

        # Calculate Inverse Concept Frequency (ICF)
        # self.doc_count_wikipedia is the total number of concepts (Wikipedia articles)
        self.icf = {
            cid: log(self.doc_count_wikipedia / freq) if freq > 0 and self.doc_count_wikipedia > 0 else 0.0
            for cid, freq in concept_freq.items()
        }
        
        # Save precomputed data
        try:
            with open(precompute_path, 'wb') as f:
                pickle.dump({
                    'concept_norms': self.concept_norms,
                    'icf': self.icf
                }, f)
            print(f"NESA parameters computed and saved to {precompute_path} in {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"Warning: Could not save precomputed NESA parameters to {precompute_path}: {e}")


    def load_wikipedia_index(self, file_path, precompute_path):
        """
        Load the pre-built Wikipedia term-to-concept index.
        Also loads or computes NESA parameters (concept_norms, icf).

        Parameters
        ----------
        file_path : str
            Path to the pickled Wikipedia index file.
        precompute_path : str
            Path to load/store precomputed concept norms and ICF.
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            self.wikipedia_index = data['index']  # term -> list of (concept_id, tf_idf_score)
            self.idf = data['idf']  # term -> idf_score (from Wikipedia corpus)
            self.doc_count_wikipedia = data['doc_count'] # Total number of Wikipedia articles/concepts
            self.concept_ids = list(range(self.doc_count_wikipedia)) # Assuming concept_ids are 0 to N-1
            self.concept_id_to_title = data.get('concept_id_to_title', {}) # Optional: mapping from concept ID to Wikipedia article title
            print(f"Successfully loaded Wikipedia index from {file_path}")
            print(f"Wikipedia index contains {self.doc_count_wikipedia} concepts and {len(self.wikipedia_index)} unique terms.")

        except FileNotFoundError:
            raise FileNotFoundError(f"Wikipedia index file {file_path} not found. Please ensure it's correctly generated and the path is correct.")
        except KeyError as e:
            raise KeyError(f"Wikipedia index file {file_path} is missing a required key: {e}. Ensure it contains 'index', 'idf', and 'doc_count'.")
        except Exception as e:
            raise Exception(f"Error loading Wikipedia index from {file_path}: {e}")

        # Load or compute NESA parameters (concept_norms, icf)
        if self.model_type == "nesa":
            if os.path.exists(precompute_path):
                try:
                    with open(precompute_path, 'rb') as f:
                        nesa_data = pickle.load(f)
                    self.concept_norms = nesa_data['concept_norms']
                    self.icf = nesa_data['icf']
                    print(f"Loaded precomputed NESA parameters from {precompute_path}")
                except Exception as e:
                    print(f"Warning: Could not load precomputed NESA parameters from {precompute_path}: {e}. Recomputing...")
                    self._compute_and_save_nesa_params(precompute_path)
            else:
                print(f"Precomputed NESA parameters not found at {precompute_path}. Computing...")
                self._compute_and_save_nesa_params(precompute_path)
        else: # For "esa", norms are effectively 1, icf is not used in score modification
            self.concept_norms = defaultdict(lambda: 1.0)
            self.icf = defaultdict(lambda: 1.0) # Effectively multiplying by 1

        # Verify key terms (optional debugging)
        # key_terms = ['aeroelast', 'aerodynam', 'slipstream', 'lift']
        # for term in key_terms:
        #     print(f"Term '{term}' in index: {term in self.wikipedia_index}, TF-IDF entries: {len(self.wikipedia_index.get(term, []))}")


    def _create_concept_vector(self, term_frequencies):
        """
        Helper function to create an ESA/NESA concept vector from term frequencies.

        Parameters
        ----------
        term_frequencies : collections.Counter
            A Counter object mapping terms to their frequencies in the text (doc or query).

        Returns
        -------
        tuple: (defaultdict(float), float)
            A tuple containing the concept vector (sparse) and its L2 norm.
        """
        concept_vector = defaultdict(float)
        vector_norm_sq = 0.0

        if not term_frequencies:
            return concept_vector, 1.0

        for term, freq in term_frequencies.items():
            if term in self.wikipedia_index:
                # Using log-weighted term frequency (TF)
                tf_weight = log(1 + freq)
                
                # For each concept associated with the term in Wikipedia
                for concept_id, term_concept_strength in self.wikipedia_index[term]:
                    score = tf_weight * term_concept_strength # Basic ESA score component

                    if self.model_type == "nesa":
                        # Adjust score for NESA: multiply by ICF and divide by concept norm
                        # Ensure concept_id exists in self.icf and self.concept_norms
                        # .get with default 1.0 for norms if a concept somehow isn't in precomputed norms
                        # .get with default 0.0 for icf (log(1)=0, so icf of 1.0 is like no effect if not found)
                        # However, _compute_and_save_nesa_params should cover all concepts in wikipedia_index
                        icf_val = self.icf.get(concept_id, 0.0) # Default to 0 if not found (conservative)
                        norm_val = self.concept_norms.get(concept_id, 1.0) # Default to 1.0 if not found
                        
                        score *= icf_val 
                        if norm_val > 1e-9: # Avoid division by zero or very small number
                           score /= norm_val
                        else:
                           score = 0 # Or handle as appropriate if norm is effectively zero

                    concept_vector[concept_id] += score
        
        # Calculate L2 norm of the constructed concept vector
        # This norm is for the *document's* or *query's* ESA vector, not the Wikipedia concept's norm
        for val in concept_vector.values():
            vector_norm_sq += val ** 2
        
        final_norm = sqrt(vector_norm_sq) if vector_norm_sq > 0 else 1.0
        return concept_vector, final_norm

    def buildIndex(self, docs, docIDs):
        """
        Build concept vectors for input documents using the Wikipedia concept space.

        Parameters
        ----------
        docs : list
            A list of lists of lists, where each sub-list is a document,
            and each sub-sub-list is a sentence of tokens.
        docIDs : list
            A list of strings denoting IDs of the documents.
        """
        print(f"Building ESA/NESA index for {len(docIDs)} documents...")
        start_time = time.time()
        self.doc_vectors = {} # Resetting doc_vectors
        self.doc_term_freq = {}

        # First, build TF-IDF index for the input documents (for hybrid scoring)
        # This assumes InformationRetrieval().buildIndex can handle the same doc format
        print("Building TF-IDF index for input documents (for hybrid scoring)...")
        self.tfidf_retriever.buildIndex(docs, docIDs)
        print("TF-IDF index for input documents built.")

        for i, (doc_content, doc_id) in enumerate(zip(docs, docIDs)):
            # Flatten document tokens and convert to lowercase, keeping only alphanumeric
            tokens = [
                token.lower() 
                for sentence in doc_content 
                for token in sentence 
                if token.isalnum() # Basic filtering
            ]
            
            if not tokens:
                # Store empty vector and norm 1.0 if doc has no valid tokens
                self.doc_vectors[doc_id] = (defaultdict(float), 1.0)
                self.doc_term_freq[doc_id] = Counter()
                continue

            term_counts = Counter(tokens)
            self.doc_term_freq[doc_id] = term_counts
            
            doc_concept_vector, doc_concept_vector_norm = self._create_concept_vector(term_counts)
            self.doc_vectors[doc_id] = (doc_concept_vector, doc_concept_vector_norm)

            if (i + 1) % 100 == 0: # Print progress every 100 documents
                print(f"  Processed {i+1}/{len(docIDs)} documents for ESA/NESA index...")

        print(f"ESA/NESA index build time: {time.time() - start_time:.2f} seconds")
        print(f"Indexed {len(self.doc_vectors)} documents into ESA/NESA vectors.")

    def rank(self, queries, alpha=0.7, beta_prf=0, gamma_expansion=0, K_prf=0, M_prf=0):
        """
        Rank documents by relevance to each query using ESA or NESA with optional TF-IDF hybrid scoring
        and pseudo-relevance feedback.

        Parameters
        ----------
        queries : list
            A list of lists of lists (query -> sentences -> tokens).
        alpha : float
            Weight for ESA/NESA score in hybrid model (0 to 1). TF-IDF weight will be (1-alpha).
        beta_prf : float
            Weight for terms from pseudo-relevance feedback during term selection.
        gamma_expansion : float
            TF weight for expanded terms added to the query vector.
        K_prf : int
            Number of top documents to use for pseudo-relevance feedback.
        M_prf : int
            Number of top terms to extract from feedback documents.


        Returns
        -------
        list
            A list of lists of document IDs, ranked by relevance for each query.
        """
        print(f"Ranking {len(queries)} queries using {self.model_type.upper()}...")
        start_time = time.time()
        
        all_ranked_doc_ids = []

        # Get TF-IDF rankings for all queries at once from the InformationRetrieval module
        # This assumes self.tfidf_retriever has been indexed with the target documents
        print("Getting TF-IDF rankings for hybrid scoring...")
        tfidf_ranked_results_all_queries = self.tfidf_retriever.rank(queries)
        print("TF-IDF rankings obtained.")

        for query_idx, query_content in enumerate(queries):
            query_tokens = [
                token.lower() 
                for sentence in query_content 
                for token in sentence 
                if token.isalnum()
            ]

            if not query_tokens:
                all_ranked_doc_ids.append([])
                print(f"Query {query_idx+1} is empty after processing. Skipping.")
                continue

            query_term_counts = Counter(query_tokens)
            
            # 1. Initial Query ESA/NESA Vector
            # Boost original query terms slightly more (e.g., by adjusting TF weight or a multiplier)
            # Original code used tf * 1.5, let's make it a parameter if needed, or adjust log(1+freq)
            # For now, let's use a simple TF for query terms: log(1 + freq)
            initial_query_concept_vector, initial_query_norm = self._create_concept_vector(query_term_counts)

            if not initial_query_concept_vector: # Empty query vector
                # If initial query vector is empty, rely solely on TF-IDF or return empty
                print(f"  Query {query_idx+1} resulted in an empty ESA/NESA vector. Using TF-IDF only for this query.")
                # tfidf_doc_ids_for_this_query will be used directly if alpha is 0 or vector is empty
                # If hybrid, this query will have 0 ESA score.
                # Fallback to TF-IDF if ESA vector is empty
                current_tfidf_ranked_doc_ids = tfidf_ranked_results_all_queries[query_idx]
                scores = {doc_id: (1.0 / (rank + 1)) for rank, doc_id in enumerate(current_tfidf_ranked_doc_ids)}
                
                # Sort all documents based on these TF-IDF scores
                # Ensure all doc IDs from self.doc_vectors are considered, giving 0 to those not in TF-IDF results
                final_scores_for_query = {doc_id: scores.get(doc_id, 0.0) for doc_id in self.doc_vectors.keys()}
                
                ranked_docs_for_query = sorted(final_scores_for_query.items(), key=lambda item: item[1], reverse=True)
                all_ranked_doc_ids.append([doc_id for doc_id, _ in ranked_docs_for_query])
                continue


            # 2. Initial Ranking based on initial query vector
            initial_esa_scores = {}
            for doc_id, (doc_concept_vector, doc_norm) in self.doc_vectors.items():
                dot_product = 0.0
                # Iterate over the smaller dictionary's keys for sparse dot product
                vec1, vec2 = initial_query_concept_vector, doc_concept_vector
                if len(doc_concept_vector) < len(initial_query_concept_vector):
                    vec1, vec2 = doc_concept_vector, initial_query_concept_vector
                
                for concept_id in vec1:
                    if concept_id in vec2:
                        dot_product += vec1[concept_id] * vec2[concept_id]
                
                # Cosine similarity
                denominator = initial_query_norm * doc_norm
                score = dot_product / denominator if denominator > 1e-9 else 0.0
                initial_esa_scores[doc_id] = score
            
            # 3. Pseudo-Relevance Feedback (PRF)
            # Sort documents by initial ESA scores to find top K for feedback
            # Ensure K_prf is not larger than the number of documents with non-zero scores
            relevant_docs_for_feedback = [
                (doc_id, score) for doc_id, score in initial_esa_scores.items() if score > 0
            ]
            
            # If K_prf is 0 or M_prf is 0, skip PRF
            expanded_query_concept_vector = initial_query_concept_vector.copy()
            current_query_norm = initial_query_norm # Will be updated if vector changes

            if K_prf > 0 and M_prf > 0 and relevant_docs_for_feedback:
                # Sort available relevant docs and take min(K_prf, len(relevant_docs_for_feedback))
                top_k_docs_for_feedback = sorted(relevant_docs_for_feedback, key=lambda x: x[1], reverse=True)[:K_prf]

                feedback_terms_scores = Counter()
                if top_k_docs_for_feedback:
                    print(f"  Performing PRF with top {len(top_k_docs_for_feedback)} docs for query {query_idx+1}.")
                    for doc_id, _ in top_k_docs_for_feedback:
                        # doc_term_freq should have been populated during buildIndex
                        doc_terms = self.doc_term_freq.get(doc_id, Counter())
                        for term, freq in doc_terms.items():
                            if term in self.wikipedia_index: # Only consider terms known to ESA
                                # Weight feedback terms by their TF in feedback doc and IDF in Wikipedia
                                # Original code used: tf * self.idf.get(term, 0.0) * 0.7
                                # self.idf is from Wikipedia, not the Cranfield corpus. This is fine.
                                term_tf_in_feedback_doc = log(1 + freq)
                                term_idf_in_wikipedia = self.idf.get(term, 0.0) # IDF from Wikipedia index
                                feedback_terms_scores[term] += term_tf_in_feedback_doc * term_idf_in_wikipedia * beta_prf
                    
                    top_m_feedback_terms = [term for term, score in feedback_terms_scores.most_common(M_prf)]
                    print(f"    Top {len(top_m_feedback_terms)} feedback terms: {top_m_feedback_terms}")

                    if top_m_feedback_terms:
                        # Create concept vector for these feedback terms
                        # The weight for these terms is gamma_expansion (original code used tf=0.4)
                        feedback_term_counts = Counter({term: gamma_expansion for term in top_m_feedback_terms})
                        feedback_concept_vector, _ = self._create_concept_vector(feedback_term_counts) 
                        
                        # Add to the initial query vector (Rocchio-like update)
                        for concept_id, score in feedback_concept_vector.items():
                            expanded_query_concept_vector[concept_id] += score # Add feedback, don't just replace
                        
                        # Recalculate norm for the expanded query vector
                        expanded_norm_sq = sum(v**2 for v in expanded_query_concept_vector.values())
                        current_query_norm = sqrt(expanded_norm_sq) if expanded_norm_sq > 0 else 1.0
            else:
                pass


            # 4. Final Ranking with Hybrid Scoring
            final_scores_for_query = {}
            
            # TF-IDF scores for the current query (already computed)
            # Convert TF-IDF ranking to scores (e.g., inverse rank or normalized scores)
            current_tfidf_ranked_doc_ids = tfidf_ranked_results_all_queries[query_idx]
            tfidf_scores_map = {
                doc_id: 1.0 / (rank + 1) # Simple inverse rank score
                for rank, doc_id in enumerate(current_tfidf_ranked_doc_ids)
            }

            # Max TF-IDF score for normalization (optional, if needed)
            # max_tfidf_score = 1.0 if tfidf_scores_map else 0.0

            for doc_id, (doc_concept_vector, doc_norm) in self.doc_vectors.items():
                # Calculate ESA/NESA score with the (potentially expanded) query vector
                dot_product = 0.0
                vec1, vec2 = expanded_query_concept_vector, doc_concept_vector
                if len(doc_concept_vector) < len(expanded_query_concept_vector):
                    vec1, vec2 = doc_concept_vector, expanded_query_concept_vector
                
                for concept_id in vec1:
                    if concept_id in vec2:
                        dot_product += vec1[concept_id] * vec2[concept_id]
                
                denominator = current_query_norm * doc_norm
                esa_nesa_score = dot_product / denominator if denominator > 1e-9 else 0.0
                
                # Get TF-IDF score
                # If a doc wasn't in TF-IDF ranking, it gets 0.
                # This assumes doc_id is string, ensure consistency.
                tfidf_score = tfidf_scores_map.get(str(doc_id), 0.0) 
                
                # Hybrid score
                hybrid_score = (alpha * esa_nesa_score) + ((1 - alpha) * tfidf_score)
                final_scores_for_query[doc_id] = hybrid_score
            
            # Sort documents by final hybrid score
            ranked_docs_for_query = sorted(final_scores_for_query.items(), key=lambda item: item[1], reverse=True)
            
            # Extract just the document IDs in order
            final_ordered_doc_ids = [str(doc_id) for doc_id, score in ranked_docs_for_query]
            all_ranked_doc_ids.append(final_ordered_doc_ids)

            # Optional: Print debug info for the query
            # print(f"  Query {query_idx+1} - Top 5 Docs: {final_ordered_doc_ids[:5]}")
            # print(f"  Query vector sparsity: {len(expanded_query_concept_vector)}/{self.doc_count_wikipedia} concepts")
            # sample_concepts = list(expanded_query_concept_vector.keys())[:5]
            # print(f"  Sample concepts in query vector: {[self.concept_id_to_title.get(c, c) for c in sample_concepts]}")


        print(f"\nTotal ranking time for {len(queries)} queries: {time.time() - start_time:.2f} seconds")
        return all_ranked_doc_ids

