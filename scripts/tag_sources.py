"""Recover which original dataset each WELFake article came from.

WELFake merges ISOT (Reuters + fake sites), McIntire, Kaggle (UTK) and BuzzFeed but ships no
source column. We match each article's normalised text prefix against ISOT and McIntire;
whatever is left is tagged `kaggle` (its titles carry the scraper's "- The New York Times" /
"- Breitbart" suffixes). Writes data/welfake_sources.csv (row_id, source).

Matching also re-confirms the label audit: every ISOT True.csv article has WELFake label 0.
"""
import sys
from pathlib import Path

import kagglehub
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fnd.data import DATA_DIR, text_key, welfake_csv  # noqa: E402

MIN_KEY = 60  # very short texts match too many articles to be informative


def keys(df):
    k = df['text'].map(text_key)
    return set(k[k.str.len() >= MIN_KEY])


def main():
    isot_dir = Path(kagglehub.dataset_download('clmentbisaillon/fake-and-real-news-dataset'))
    isot = {f: pd.read_csv(isot_dir / f) for f in ['True.csv', 'Fake.csv']}
    mcintire = pd.read_csv(next(Path(kagglehub.dataset_download('hassanamin/textdb3')).rglob('*.csv')))

    w = pd.read_csv(welfake_csv()).rename(columns={'Unnamed: 0': 'row_id'})
    k = w['text'].map(text_key)
    w['source'] = 'kaggle'
    w.loc[k.isin(keys(mcintire)), 'source'] = 'mcintire'
    w.loc[k.isin(keys(pd.concat(isot.values()))), 'source'] = 'isot'

    print(pd.crosstab(w['source'], w['label']))
    for f, d in isot.items():
        print(f'ISOT {f}: WELFake label counts', w.loc[k.isin(keys(d)), 'label'].value_counts().to_dict())

    DATA_DIR.mkdir(exist_ok=True)
    w[['row_id', 'source']].to_csv(DATA_DIR / 'welfake_sources.csv', index=False)


if __name__ == '__main__':
    main()
