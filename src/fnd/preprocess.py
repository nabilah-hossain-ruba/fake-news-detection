"""Text views for each model family, cached because lemmatising 60k articles is slow.

- `classical`: the original notebook pipeline (lowercase, strip URLs/HTML/punctuation/digits,
  remove stopwords, lemmatise) for TF-IDF models.
- `light`: lowercase word tokens with stopwords kept, for the BiLSTM.
- `raw`: title + text as written, for DistilBERT (its tokenizer handles casing and punctuation).
Each view exists for the `original` and the artifact-`stripped` corpus.
"""
import hashlib
import pickle
import re
import string

import nltk

from .artifacts import BODY_PATTERNS, TITLE_PATTERNS, strip_artifacts
from .data import CACHE_DIR

_STOP, _LEM = None, None


def _nltk():
    global _STOP, _LEM
    if _STOP is None:
        for r in ['stopwords', 'punkt', 'punkt_tab', 'wordnet']:
            nltk.download(r, quiet=True)
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        _STOP, _LEM = set(stopwords.words('english')), WordNetLemmatizer()
    return _STOP, _LEM


def clean_classical(text):
    stop, lem = _nltk()
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = nltk.word_tokenize(text)
    return ' '.join(lem.lemmatize(t) for t in tokens if t not in stop and len(t) > 2)


def clean_light(text):
    text = re.sub(r'http\S+|www\.\S+|<.*?>', ' ', str(text).lower())
    return ' '.join(re.findall(r"[a-z]+(?:'[a-z]+)?", text))


def _raw(df, condition):
    titles, texts = df['title'].astype(str).tolist(), df['text'].astype(str).tolist()
    if condition == 'stripped':
        titles, texts = map(list, zip(*(strip_artifacts(t, x) for t, x in zip(titles, texts))))
    elif condition != 'original':
        raise ValueError(condition)
    return [f'{t}. {x}' for t, x in zip(titles, texts)]


_CLEANERS = {'raw': None, 'light': clean_light, 'classical': clean_classical}


def text_view(df, condition, view):
    """list[str] aligned with df: `view` of the `condition` ('original' or 'stripped') corpus."""
    # Key the cache on the artifact patterns so editing them invalidates stripped views
    tag = hashlib.md5(repr((BODY_PATTERNS, TITLE_PATTERNS)).encode()).hexdigest()[:8]
    path = CACHE_DIR / f'{view}_{condition}{"_" + tag if condition == "stripped" else ""}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            texts = pickle.load(f)
        if len(texts) == len(df):
            return texts

    texts = _raw(df, condition)
    if _CLEANERS[view] is not None:
        texts = [_CLEANERS[view](t) for t in texts]
    CACHE_DIR.mkdir(exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(texts, f)
    return texts
