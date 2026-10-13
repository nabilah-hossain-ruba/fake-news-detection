"""Loading, de-duplication, source tags and splits for WELFake."""
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data'
# Overridable so Colab runs can keep cache and results on Google Drive and resume after a disconnect
CACHE_DIR = Path(os.environ.get('FND_CACHE_DIR', ROOT / 'cache'))
RESULTS_DIR = Path(os.environ.get('FND_RESULTS_DIR', ROOT / 'results'))

# The Kaggle page says 0 = fake, 1 = real; the data says the opposite (see README, "Label audit").
LABEL_NAMES = ['Real', 'Fake']
SOURCES = ['isot', 'mcintire', 'kaggle']
SOURCE_NAMES = {'isot': 'ISOT (Reuters + fake sites)', 'mcintire': 'McIntire', 'kaggle': 'Kaggle (UTK)'}


def welfake_csv():
    """Find WELFake_Dataset.csv locally, or download it with kagglehub."""
    candidates = [os.environ.get('WELFAKE_CSV'),
                  DATA_DIR / 'WELFake_Dataset.csv',
                  ROOT / 'WELFAKE dataset_FAKE NEWS DETECTION' / 'WELFake_Dataset.csv']
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    import kagglehub
    path = Path(kagglehub.dataset_download('saurabhshahane/fake-news-classification'))
    return next(path.glob('*.csv'))


def text_key(s, n=300):
    """Normalised prefix of an article, used to match the same article across datasets."""
    return re.sub(r'\W+', '', str(s).lower())[:n]


def _shingles(text, k=5, max_words=1500):
    words = re.findall(r'\w+', str(text).lower())[:max_words]
    return {' '.join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}


def near_duplicate_groups(texts, threshold=0.8, num_perm=64, seed=0):
    """Group ids from MinHash-LSH on word 5-shingles (Jaccard >= ~threshold)."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, t in enumerate(texts):
        m = MinHash(num_perm=num_perm, seed=seed)
        m.update_batch([s.encode('utf8') for s in _shingles(t)])
        for j in lsh.query(m):
            parent[find(i)] = find(j)
        lsh.insert(i, m)
    return np.array([find(i) for i in range(len(texts))])


def _drop_duplicate_groups(df, groups):
    """Keep the first article per group; drop whole groups whose labels disagree."""
    df = df.assign(_g=groups)
    conflicting = df.groupby('_g')['label'].transform('nunique') > 1
    kept = df[~conflicting].drop_duplicates('_g')
    return kept.drop(columns='_g'), int(conflicting.sum()), len(df) - int(conflicting.sum()) - len(kept)


def load_corpus(use_cache=True):
    """Cleaned, de-duplicated WELFake with a `source` column.

    Steps: drop rows with missing title/text/label or empty text, attach source tags,
    drop exact duplicates (normalised text), then near-duplicates (MinHash-LSH).
    Groups whose copies carry different labels are removed entirely.
    Returns (df, stats) where stats records how many rows each step removed.
    """
    cache = CACHE_DIR / 'corpus.pkl'
    if use_cache and cache.exists():
        with open(cache, 'rb') as f:
            return pickle.load(f)

    raw = pd.read_csv(welfake_csv()).rename(columns={'Unnamed: 0': 'row_id'})
    stats = {'raw': len(raw)}

    df = raw.dropna(subset=['title', 'text', 'label'])
    df = df[df['text'].astype(str).str.strip().str.len() > 0]
    stats['after_missing_empty'] = len(df)

    sources = pd.read_csv(DATA_DIR / 'welfake_sources.csv')
    df = df.merge(sources, on='row_id', how='left')
    df['source'] = df['source'].fillna('kaggle')

    df['key'] = df['text'].map(lambda s: re.sub(r'\W+', '', str(s).lower()))
    df, conflict, dup = _drop_duplicate_groups(df, df['key'].factorize()[0])
    stats['exact_dup_removed'], stats['exact_conflict_removed'] = dup, conflict

    df = df.reset_index(drop=True)
    df, conflict, dup = _drop_duplicate_groups(df, near_duplicate_groups(df['text'].tolist()))
    stats['near_dup_removed'], stats['near_conflict_removed'] = dup, conflict

    df = df.drop(columns='key').reset_index(drop=True)
    df['label'] = df['label'].astype(int)
    stats['final'] = len(df)

    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache, 'wb') as f:
        pickle.dump((df, stats), f)
    return df, stats


def in_domain_split(df, seed, val_size=0.1, test_size=0.2):
    """Stratified (label x source) train/val/test indices."""
    strata = df['label'].astype(str) + '_' + df['source']
    idx = np.arange(len(df))
    train_val, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strata)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size),
                                  random_state=seed, stratify=strata.iloc[train_val])
    return {'train': train, 'val': val, 'test': test}


def cross_source_split(df, test_source, seed, val_size=0.1):
    """Train/val on the other sources; test on every article from `test_source`."""
    idx = np.arange(len(df))
    held_out = (df['source'] == test_source).to_numpy()
    rest = idx[~held_out]
    strata = (df['label'].astype(str) + '_' + df['source']).iloc[rest]
    train, val = train_test_split(rest, test_size=val_size, random_state=seed, stratify=strata)
    return {'train': train, 'val': val, 'test': idx[held_out]}


def iter_splits(df, setting, seeds):
    """Yield (split_name, seed, split) for an experimental setting."""
    for seed in seeds:
        if setting == 'in-domain':
            yield 'in-domain', seed, in_domain_split(df, seed)
        elif setting == 'cross-source':
            for src in SOURCES:
                yield f'heldout-{src}', seed, cross_source_split(df, src, seed)
        else:
            raise ValueError(setting)
