"""Where runs write predictions and metadata, so analysis can be redone from disk."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import RESULTS_DIR

PRED_DIR = RESULTS_DIR / 'preds'
# One runs_<family>_<setting>.jsonl per model family and setting, so logs from separate
# machines (Kaggle, Colab, local) can be copied into one results/ folder without clashing
RUNS_DIR = RESULTS_DIR


def pred_path(split_name, train_cond, test_cond, model, seed):
    return PRED_DIR / split_name / f'{train_cond}__{test_cond}' / model.replace(' ', '_') / f'seed{seed}.csv.gz'


def done(split_name, train_cond, test_cond, model, seed):
    return pred_path(split_name, train_cond, test_cond, model, seed).exists()


def save_predictions(split_name, train_cond, test_cond, model, seed, row_ids, y_true, y_pred, y_score):
    path = pred_path(split_name, train_cond, test_cond, model, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'row_id': row_ids, 'y_true': y_true, 'y_pred': np.asarray(y_pred).astype(int),
                  'y_score': y_score}).to_csv(path, index=False)


def log_run(**record):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    setting = 'in-domain' if record['split'] == 'in-domain' else 'cross-source'
    with open(RUNS_DIR / f"runs_{record['family']}_{setting}.jsonl", 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, default=str) + '\n')


def load_runs():
    lines = [l for p in sorted(RUNS_DIR.glob('runs_*.jsonl')) for l in p.read_text(encoding='utf-8').splitlines()]
    return pd.DataFrame([json.loads(l) for l in lines])


def load_predictions():
    """All prediction files as one long frame with split/condition/model/seed columns."""
    frames = []
    for path in PRED_DIR.glob('*/*/*/seed*.csv.gz'):
        split_name, conds, model = path.parts[-4:-1]
        train_cond, test_cond = conds.split('__')
        frames.append(pd.read_csv(path).assign(split=split_name, train_cond=train_cond, test_cond=test_cond,
                                              model=model.replace('_', ' '), seed=int(Path(path.stem).stem[4:])))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
