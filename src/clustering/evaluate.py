# evaluate.py
# scores to compare the clusterings, no labels used here

import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def score_labels(X, labels):
    # silhouette is very slow 3la 53k rows, so we take a 10k sample
    sil = silhouette_score(X, labels, sample_size=10000, random_state=42)
    db = davies_bouldin_score(X, labels)     # lower is better
    ch = calinski_harabasz_score(X, labels)  # higher is better

    # size of the smallest cluster (0.03 ya3ni 3%)
    shares = pd.Series(labels).value_counts(normalize=True)

    return {
        "silhouette": float(sil),
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
        "n_clusters": len(shares),
        "min_cluster_share": float(shares.min()),
    }


def pick_best(search_table, min_share=0.03):
    # sheel ay setting el smallest cluster fih less than 3%
    good = search_table[search_table["min_cluster_share"] >= min_share]

    # if all of them are too small, just use the full table
    if len(good) == 0:
        good = search_table

    # highest silhouette first, w law fi tie the lower davies_bouldin wins
    good = good.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True])
    return good.iloc[0]