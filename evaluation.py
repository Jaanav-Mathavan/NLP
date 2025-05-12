from collections import defaultdict
import numpy as np

class Evaluation:
    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Compute precision at k for a single query
        """
        retrieved = query_doc_IDs_ordered[:k]
        relevant = [doc_id for doc_id in retrieved if doc_id in true_doc_IDs]
        return len(relevant) / k

    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Compute mean precision at k over all queries
        """
        precision_list = []
        for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
            query_id = str(query_id)
            rel_doc_ids = [x["id"] for x in qrels if x["query_num"] == query_id]
            if not rel_doc_ids:
                print(f"No relevant docs for query {query_id}")
            q_precision = self.queryPrecision(doc_id_order, query_id, rel_doc_ids, k)
            precision_list.append(q_precision)
        mean_precision = sum(precision_list) / len(precision_list) if precision_list else 0.0
        print(f"Mean Precision@{k}: {mean_precision:.4f}")
        return mean_precision

    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Compute recall at k for a single query
        """
        retrieved = query_doc_IDs_ordered[:k]
        relevant = [doc_id for doc_id in retrieved if doc_id in true_doc_IDs]
        return len(relevant) / len(true_doc_IDs) if true_doc_IDs else 0.0

    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Compute mean recall at k over all queries
        """
        recall_list = []
        for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
            query_id = str(query_id)
            rel_doc_ids = [x["id"] for x in qrels if x["query_num"] == query_id]
            q_recall = self.queryRecall(doc_id_order, query_id, rel_doc_ids, k)
            recall_list.append(q_recall)
        mean_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
        print(f"Mean Recall@{k}: {mean_recall:.4f}")
        return mean_recall

    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Compute F-score at k for a single query
        """
        alpha = 0.5
        precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        return (1 + alpha**2) * (precision * recall) / ((alpha**2 * precision) + recall) if (precision + recall) > 0 else 0.0

    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Compute mean F-score at k over all queries
        """
        fscore_list = []
        for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
            query_id = str(query_id)
            rel_doc_ids = [x["id"] for x in qrels if x["query_num"] == query_id]
            q_fscore = self.queryFscore(doc_id_order, query_id, rel_doc_ids, k)
            fscore_list.append(q_fscore)
        mean_fscore = sum(fscore_list) / len(fscore_list) if fscore_list else 0.0
        print(f"Mean F-score@{k}: {mean_fscore:.4f}")
        return mean_fscore

    def queryNDCG(self, query_doc_IDs_ordered, query_id, qrels, k):
        """
        Compute nDCG at k for a single query
        """
        true_docs = {x["id"]: 1 for x in qrels if x["query_num"] == query_id}
        dcg = 0
        for index, retrieved_doc in enumerate(query_doc_IDs_ordered[:k]):
            if retrieved_doc in true_docs:
                dcg += true_docs[retrieved_doc] / np.log2(index + 2)
        ideal_dcg = 0
        sorted_docs = sorted(true_docs.keys(), key=lambda x: true_docs[x], reverse=True)[:k]
        for index, true_doc in enumerate(sorted_docs):
            ideal_dcg += true_docs[true_doc] / np.log2(index + 2)
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Compute mean nDCG at k over all queries
        """
        nDCG_list = []
        for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
            query_id = str(query_id)
            q_nDCG = self.queryNDCG(doc_id_order, query_id, qrels, k)
            nDCG_list.append(q_nDCG)
        mean_nDCG = sum(nDCG_list) / len(nDCG_list) if nDCG_list else 0.0
        print(f"Mean nDCG@{k}: {mean_nDCG:.4f}")
        return mean_nDCG

    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Compute average precision at k for a single query
        """
        rel_precision_list = []
        for index, doc_id in enumerate(query_doc_IDs_ordered[:k]):
            if doc_id in true_doc_IDs:
                q_precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, index + 1)
                rel_precision_list.append(q_precision)
        return sum(rel_precision_list) / len(rel_precision_list) if rel_precision_list else 0.0

    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Compute MAP at k over all queries
        """
        avg_precision_list = []
        for query_id, doc_id_order in zip(query_ids, doc_IDs_ordered):
            query_id = str(query_id)
            rel_doc_ids = [x["id"] for x in qrels if x["query_num"] == query_id]
            qavg_precision = self.queryAveragePrecision(doc_id_order, query_id, rel_doc_ids, k)
            avg_precision_list.append(qavg_precision)
        mean_ap = sum(avg_precision_list) / len(avg_precision_list) if avg_precision_list else 0.0
        print(f"Mean AP@{k}: {mean_ap:.4f}")
        return mean_ap
