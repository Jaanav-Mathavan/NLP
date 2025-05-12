import pandas as pd
import re
from scipy.stats import ttest_rel

def find_tfidf_column(columns):
    norm = lambda c: re.sub(r'[^a-z]', '', c.lower())
    for c in columns:
        if norm(c) == 'tfidf':
            return c
    raise KeyError(f"Could not find a TF–IDF column in {list(columns)}")

files = {
    'MAP@10':    '/home/meanAveragePrecision.csv',
    'Precision@10': '/home/meanPrecision.csv',
    'Recall@10':    '/home/meanRecall.csv',
    'nDCG@10':      '/home/meanNDCG.csv',
    'F1@10':        '/home/meanFscore.csv'
}

results = []

for metric, path in files.items():
    df = pd.read_csv(path, index_col=0)
    try:
        tfidf_col = find_tfidf_column(df.columns)
    except KeyError as e:
        print(f"[ERROR] {e}")
        print("Columns found:", df.columns.tolist())
        continue

    baseline = df[tfidf_col]
    for model in df.columns:
        if model == tfidf_col:
            continue
        t_stat, p_val = ttest_rel(baseline, df[model])
        results.append({
            'Metric': metric,
            'Baseline': tfidf_col,
            'Model': model,
            't-stat': round(t_stat, 4),
            'p-value': f"{p_val:.3e}",
            'Significant (α=0.05)': 'Yes' if p_val < 0.05 else 'No'
        })

# Display results
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
