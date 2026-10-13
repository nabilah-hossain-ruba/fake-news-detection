"""Metrics, bootstrap confidence intervals and McNemar tests."""
import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def metrics(y_true, y_pred, y_score):
    return {'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else np.nan}


def bootstrap_ci(y_true, y_pred, metric=f1_score, n=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a metric over test articles."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    stats = [metric(y_true[i], y_pred[i]) for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def mcnemar(y_true, pred_a, pred_b):
    """McNemar test on paired predictions (exact binomial when discordant pairs < 25)."""
    y_true, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    only_a = int(np.sum((a == y_true) & (b != y_true)))
    only_b = int(np.sum((a != y_true) & (b == y_true)))
    if only_a + only_b == 0:
        return 1.0
    if only_a + only_b < 25:
        return float(binomtest(only_a, only_a + only_b, 0.5).pvalue)
    stat = (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b)
    return float(chi2.sf(stat, 1))


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted
