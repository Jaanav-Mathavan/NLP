import pandas as pd
import re
from scipy.stats import ttest_rel

# Helper to find the TF–IDF column (case‐insensitive)
def find_tfidf_column(columns):
    norm = lambda c: re.sub(r'[^a-z]', '', c.lower())
    for c in columns:
        if norm(c) == 'tfidf':
            return c
    raise KeyError(f"Could not find a TF–IDF column in {list(columns)}")

# Paths to your per-query CSVs
files = {
    'MAP@10':      '/home/meanAveragePrecision.csv',
    'Precision@10':'/home/meanPrecision.csv',
    'Recall@10':   '/home/meanRecall.csv',
    'nDCG@10':     '/home/meanNDCG.csv',
    'F1@10':       '/home/meanFscore.csv'
}

alpha = 0.05
summary = []

for metric, path in files.items():
    # 1) Load the per-query results
    df = pd.read_csv(path, index_col=0)
    tfidf_col = find_tfidf_column(df.columns)
    baseline = df[tfidf_col]
    
    # 2) Compare each model to TF–IDF
    for model in df.columns:
        if model == tfidf_col:
            continue
        
        t_stat, p_two = ttest_rel( baseline,df[model])
        # convert two‐tailed p to one‐tailed for H1: model > TF–IDF
        p_one = (p_two / 2) if t_stat > 0 else (1 - p_two/2)
        
        # 3) Decide which is better
        if p_one < alpha:
            better = model if t_stat > 0 else 'TF–IDF'
        else:
            better = 'no significant difference'
        
        summary.append({
            'Metric': metric,
            'Comparison': f'{model} vs TF–IDF',
            't-stat': round(t_stat,4),
            'p-one-tailed': f'{p_one:.3e}',
            'Better': better
        })

# 4) Display the summary
summary_df = pd.DataFrame(summary)
print(summary_df[['Metric','Comparison','t-stat','p-one-tailed','Better']].to_string(index=False))
