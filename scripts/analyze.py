"""Turn saved predictions into the paper's tables and figures (results/RESULTS.md, results/figures/)."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fnd import results  # noqa: E402
from fnd.artifacts import artifact_table  # noqa: E402
from fnd.data import RESULTS_DIR, SOURCE_NAMES, load_corpus  # noqa: E402
from fnd.stats import bootstrap_ci, holm, mcnemar, metrics  # noqa: E402

FIG_DIR = RESULTS_DIR / 'figures'
TABLE_DIR = RESULTS_DIR / 'tables'
CONDS = [('original', 'original'), ('original', 'stripped'), ('stripped', 'stripped')]
COND_LABEL = {('original', 'original'): 'orig→orig', ('original', 'stripped'): 'orig→stripped',
              ('stripped', 'stripped'): 'stripped→stripped'}


def pm(mean, std):
    return f'{mean:.4f} ± {std:.4f}' if not np.isnan(std) else f'{mean:.4f}'


def md(df, index=True):
    return df.to_markdown(index=index)


def per_run_metrics(preds):
    rows = []
    for key, g in preds.groupby(['split', 'train_cond', 'test_cond', 'model', 'seed']):
        rows.append(dict(zip(['split', 'train_cond', 'test_cond', 'model', 'seed'], key),
                         n=len(g), **metrics(g['y_true'], g['y_pred'], g['y_score'])))
    return pd.DataFrame(rows)


def train_seconds():
    runs = results.load_runs()
    if runs.empty:
        return {}
    runs = runs[(runs['split'] == 'in-domain') & (runs['train_cond'] == 'original') & (runs['test_cond'] == 'original')]
    return runs.groupby('model')['train_seconds'].median().to_dict()


def data_section(df, stats):
    lines = ['## Data', '', '### Cleaning and de-duplication', '']
    steps = [('Raw WELFake rows', stats['raw']),
             ('After dropping missing title/text/label and empty text', stats['after_missing_empty']),
             ('Exact duplicates removed (normalised text)', -stats['exact_dup_removed']),
             ('Exact duplicates with conflicting labels removed', -stats['exact_conflict_removed']),
             ('Near-duplicates removed (MinHash-LSH, Jaccard ≈ 0.8)', -stats['near_dup_removed']),
             ('Near-duplicates with conflicting labels removed', -stats['near_conflict_removed']),
             ('**Final corpus**', stats['final'])]
    lines += [md(pd.DataFrame(steps, columns=['Step', 'Rows']), index=False), '']
    counts = pd.crosstab(df['source'].map(SOURCE_NAMES), df['label'].map({0: 'Real (0)', 1: 'Fake (1)'}),
                         margins=True, margins_name='Total')
    lines += ['### Source composition (recovered by `scripts/tag_sources.py`)', '', md(counts), '']
    art = artifact_table(df).set_index('artifact').rename(columns={'label_0': 'Real (0)', 'label_1': 'Fake (1)'})
    lines += ['### Share of articles containing each source artifact', '', md((art * 100).round(1)), '']
    return lines


def main_table(run_m):
    d = run_m[(run_m['split'] == 'in-domain') & (run_m['train_cond'] == 'original') & (run_m['test_cond'] == 'original')]
    secs = train_seconds()
    rows = []
    for model, g in d.groupby('model'):
        rows.append({'Model': model, 'Seeds': len(g),
                     **{k.upper() if k == 'f1' else k.replace('_', '-').title(): pm(g[k].mean(), g[k].std())
                        for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']},
                     'Train time (s, median)': round(secs.get(model, np.nan), 1)})
    table = pd.DataFrame(rows).sort_values('F1', ascending=False)
    table.to_csv(TABLE_DIR / 'in_domain_original.csv', index=False)
    return table


def significance(preds):
    """McNemar of the best model vs each other model, per seed, Holm-adjusted within seed."""
    d = preds[(preds['split'] == 'in-domain') & (preds['train_cond'] == 'original') & (preds['test_cond'] == 'original')]
    f1 = per_run_metrics(d).groupby('model')['f1'].mean()
    best = f1.idxmax()
    rows = {}
    for seed, g in d.groupby('seed'):
        wide = g.pivot_table(index='row_id', columns='model', values='y_pred')
        if best not in wide:
            continue
        truth = g.drop_duplicates('row_id').set_index('row_id')['y_true'].loc[wide.index]
        others = [m for m in wide.columns if m != best]
        adj = holm([mcnemar(truth, wide[best], wide[m]) for m in others])
        for m, p in zip(others, adj):
            rows.setdefault(m, []).append(p)
    seed0 = d[(d['seed'] == d['seed'].min())]
    out = []
    for m, ps in rows.items():
        g = seed0[seed0['model'] == m]
        lo, hi = bootstrap_ci(g['y_true'], g['y_pred'])
        out.append({'Model': m, f'Median Holm-adjusted p vs {best}': f'{np.median(ps):.2g}',
                    'Seeds with p < 0.05': f'{sum(p < 0.05 for p in ps)}/{len(ps)}',
                    'F1 95% CI (seed 0, bootstrap)': f'[{lo:.4f}, {hi:.4f}]'})
    g = seed0[seed0['model'] == best]
    lo, hi = bootstrap_ci(g['y_true'], g['y_pred'])
    out.insert(0, {'Model': f'{best} (reference)', f'Median Holm-adjusted p vs {best}': '—',
                   'Seeds with p < 0.05': '—', 'F1 95% CI (seed 0, bootstrap)': f'[{lo:.4f}, {hi:.4f}]'})
    return pd.DataFrame(out)


def ablation(run_m):
    d = run_m[run_m['split'] == 'in-domain']
    agg = d.groupby(['model', 'train_cond', 'test_cond'])['f1'].agg(['mean', 'std']).reset_index()
    table = pd.DataFrame({'Model': sorted(d['model'].unique())}).set_index('Model')
    for tr, te in CONDS:
        a = agg[(agg['train_cond'] == tr) & (agg['test_cond'] == te)].set_index('model')
        table[COND_LABEL[(tr, te)]] = [pm(*a.loc[m, ['mean', 'std']]) if m in a.index else '' for m in table.index]
    mean = agg.pivot_table(index='model', columns=['train_cond', 'test_cond'], values='mean')
    if ('original', 'stripped') in mean:
        table['Δ test-time stripping'] = (mean[('original', 'stripped')] - mean[('original', 'original')]).round(4)
    if ('stripped', 'stripped') in mean:
        table['Δ train+test stripping'] = (mean[('stripped', 'stripped')] - mean[('original', 'original')]).round(4)
    table = table.sort_values(table.columns[0], ascending=False)
    table.to_csv(TABLE_DIR / 'ablation.csv')

    plot = agg.assign(cond=[COND_LABEL[(a, b)] for a, b in zip(agg['train_cond'], agg['test_cond'])])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.barplot(data=plot, x='model', y='mean', hue='cond', hue_order=list(COND_LABEL.values()), ax=ax)
    ax.set_ylim(max(0, plot['mean'].min() - 0.1), 1)
    ax.set_xlabel(''); ax.set_ylabel('F1 (Fake class), mean over seeds')
    ax.set_title('Effect of removing source artifacts (in-domain split)')
    ax.legend(title='train → test', loc='lower right')
    plt.xticks(rotation=20, ha='right'); plt.tight_layout()
    fig.savefig(FIG_DIR / 'ablation.png', dpi=150); plt.close(fig)
    return table


def cross_source(run_m):
    d = run_m[run_m['split'].str.startswith('heldout-')]
    if d.empty:
        return None
    d = d.assign(source=d['split'].str.replace('heldout-', '').map(SOURCE_NAMES))
    indom = run_m[(run_m['split'] == 'in-domain')].groupby(['model', 'train_cond', 'test_cond'])['f1'].mean()
    tables = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (tr, te) in zip(axes, [('original', 'original'), ('stripped', 'stripped')]):
        g = d[(d['train_cond'] == tr) & (d['test_cond'] == te)]
        t = g.pivot_table(index='model', columns='source', values='f1', aggfunc='mean')
        t['In-domain F1'] = [indom.get((m, tr, te), np.nan) for m in t.index]
        t = t.sort_values('In-domain F1', ascending=False)
        tables[COND_LABEL[(tr, te)]] = t.round(4)
        t.drop(columns='In-domain F1').round(3).to_csv(TABLE_DIR / f'cross_source_{tr}.csv')
        sns.heatmap(t.drop(columns='In-domain F1'), annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.3, vmax=1,
                    cbar=ax is axes[-1], ax=ax)
        ax.set_title(f'Held-out source F1 ({COND_LABEL[(tr, te)]})'); ax.set_xlabel('Held-out test source'); ax.set_ylabel('')
    plt.tight_layout(); fig.savefig(FIG_DIR / 'cross_source.png', dpi=150); plt.close(fig)
    return tables


def explain_section():
    dirs = sorted((RESULTS_DIR / 'explain').glob('*'))
    if not dirs:
        return []
    lines = ['## DistilBERT attributions (Integrated Gradients)', '',
             'Share of each article\'s total |attribution| that falls on artifact words '
             '(`reuters`, `getty`, `featured`, `pic`, `breitbart`, `nyt`, URL pieces, `@`). '
             'The word list is deliberately narrow, so this is a lower bound.', '']
    rows = []
    for d in dirs:
        split, tr, te, seed = d.name.split('__')
        a = pd.read_csv(d / 'articles.csv')
        rows.append({'Split': split, 'Train → test': f'{tr} → {te}', 'Seed': seed[4:], 'Articles': len(a),
                     'Artifact share (Real)': round(a.loc[a['label'] == 0, 'artifact_share'].mean(), 4),
                     'Artifact share (Fake)': round(a.loc[a['label'] == 1, 'artifact_share'].mean(), 4)})
    lines += [md(pd.DataFrame(rows), index=False), '']
    first = next((d for d in dirs if d.name.startswith('in-domain__original__original')), dirs[0])
    w = pd.read_csv(first / 'words.csv')
    lines += [f'Top words by summed attribution ({first.name}); positive pushes toward Fake:', '',
              md(pd.concat([w.nlargest(15, 'sum')[['word', 'sum']].reset_index(drop=True).add_prefix('→Fake '),
                            w.nsmallest(15, 'sum')[['word', 'sum']].reset_index(drop=True).add_prefix('→Real ')],
                           axis=1).round(3), index=False), '']
    return lines


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df, stats = load_corpus()
    preds = results.load_predictions()
    if preds.empty:
        sys.exit('No predictions found under results/preds; run the experiment scripts first.')
    run_m = per_run_metrics(preds)
    run_m.to_csv(TABLE_DIR / 'per_run_metrics.csv', index=False)

    lines = ['# Results', '', '*Generated by `scripts/analyze.py`; do not edit by hand.* '
             'Label 1 = Fake is the positive class for precision, recall and F1.', '']
    lines += data_section(df, stats)
    lines += ['## In-domain results (train and test on original text)', '',
              'Stratified 70/10/20 train/val/test split per seed; hyperparameters chosen on validation.', '',
              md(main_table(run_m), index=False), '',
              '### Significance', '', md(significance(preds), index=False), '']
    lines += ['## Artifact ablation (in-domain)', '',
              '- **orig→stripped**: trained on original text, tested with artifacts removed (reliance on artifacts).',
              '- **stripped→stripped**: trained and tested without artifacts (what content alone supports).', '',
              md(ablation(run_m)), '', '![Ablation](figures/ablation.png)', '']
    cs = cross_source(run_m)
    if cs:
        lines += ['## Cross-source generalisation', '',
                  'Train on two sources, test on the third. Values are F1 (mean over seeds).', '']
        for name, t in cs.items():
            lines += [f'**{name}**', '', md(t), '']
        lines += ['![Cross-source](figures/cross_source.png)', '']
    lines += explain_section()
    (RESULTS_DIR / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', RESULTS_DIR / 'RESULTS.md')


if __name__ == '__main__':
    main()
