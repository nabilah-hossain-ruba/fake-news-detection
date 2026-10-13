"""Source-specific artifacts that reveal *where* an article was scraped, not *what* it says.

`strip_artifacts` removes them so models can be trained/tested without the shortcut.
The patterns are applied to both classes alike.
"""
import re

import pandas as pd

# (name, pattern, replacement) applied in order to the article body
BODY_PATTERNS = [
    # "WASHINGTON (Reuters) - " dateline that opens almost every Reuters article
    ('reuters_dateline', r'\A[^\n]{0,150}?\(Reuters\)\s*[-–—]\s*', ''),
    ('reuters', r'(?i)\bReuters\b', ' '),
    # Other datelines, e.g. NYT-style "WASHINGTON — " (upper-case city, then a dash)
    ('dateline', r"\A\s*[A-Z][A-Z.,' ]{2,40}\s+[-–—]{1,2}\s+", ''),
    # Social share counters at the top of scraped pages: "2 Shares 1 0 0 1"
    ('share_counter', r'\A\s*\d+\s+Shares?\b(?:\s+\d+)*', ''),
    ('read_more', r'(?i)\bRead more\s*:[^\n]*', ' '),
    # Embedded tweet signature: "- Jane Doe (@jdoe) January 1, 2017"
    ('tweet_signature', r'[—–-]\s*[^\n()]{1,60}\(@\w{1,30}\)\s+\w+\s+\d{1,2},\s+\d{4}', ' '),
    ('pic_twitter', r'(?i)pic\.twitter\.com/\S+', ' '),
    ('url', r'(?i)https?://\S+|www\.\S+', ' '),
    ('handle', r'@\w{1,30}', ' '),
    ('featured_image', r'(?i)Featured image[^\n]*', ' '),
    ('image_credit', r'(?i)\b(?:Photo|Image|Picture)s?\s+(?:credit|by|via|courtesy)\b[^\n]*', ' '),
    ('getty', r'(?i)\bGetty(?:\s+Images)?\b', ' '),
    ('21wire', r'(?i)\b21st Century Wire\b|\b21WIRE\b', ' '),
    ('outlet_names', r'(?i)\b(?:The\s+)?New York Times\b|\bNYT\b|\bBreitbart(?: News)?\b', ' '),
]

# Outlet suffixes on titles, e.g. "... - The New York Times" or "... - American Herald Tribune"
TITLE_PATTERNS = [
    ('title_outlet_suffix', r"\s+[-|]\s+(?:[A-Z][\w.&']*\s?){1,5}$", ''),
    ('outlet_names', r'(?i)\b(?:The\s+)?New York Times\b|\bNYT\b|\bBreitbart(?: News)?\b', ' '),
]

_BODY = [(n, re.compile(p, re.M), r) for n, p, r in BODY_PATTERNS]
_TITLE = [(n, re.compile(p), r) for n, p, r in TITLE_PATTERNS]


def _apply(text, patterns):
    text = str(text)
    for _, rx, rep in patterns:
        text = rx.sub(rep, text)
    return re.sub(r'[ \t]+', ' ', text).strip()


def strip_artifacts(title, text):
    return _apply(title, _TITLE), _apply(text, _BODY)


def artifact_table(df):
    """Share of articles per class that contain each artifact (before stripping)."""
    rows = []
    for name, rx, _ in _BODY:
        hit = df['text'].astype(str).map(lambda t: bool(rx.search(t)))
        rows.append({'artifact': name, **hit.groupby(df['label']).mean().rename(lambda l: f'label_{l}').to_dict()})
    for name, rx, _ in _TITLE[:1]:
        hit = df['title'].astype(str).map(lambda t: bool(rx.search(t)))
        rows.append({'artifact': name, **hit.groupby(df['label']).mean().rename(lambda l: f'label_{l}').to_dict()})
    return pd.DataFrame(rows)
