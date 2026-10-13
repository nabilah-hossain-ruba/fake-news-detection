"""Train and evaluate the TF-IDF models.

For every split and seed, models are trained on the `original` or artifact-`stripped` corpus:
  original -> tested on original and on stripped (how much does the model lean on artifacts?)
  stripped -> tested on stripped (how well can it do without them?)
Hyperparameters are chosen on the validation split. The tree ensembles are tuned on the
first seed of each split/condition and reuse those settings for later seeds to save time.
Finished runs are skipped, so the script can be stopped and restarted.

  python scripts/run_classical.py --setting in-domain --seeds 0 1 2 3 4
  python scripts/run_classical.py --setting cross-source --seeds 0
"""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fnd import classical, results  # noqa: E402
from fnd.data import iter_splits, load_corpus  # noqa: E402
from fnd.preprocess import text_view  # noqa: E402
from fnd.stats import metrics  # noqa: E402

TUNE_ONCE = {'Decision Tree', 'Random Forest', 'XGBoost'}
TEST_CONDS = {'original': ['original', 'stripped'], 'stripped': ['stripped']}


def tuned_params():
    """(split, train_cond, model) -> params from runs that did their own grid search."""
    params = {}
    for _, r in results.load_runs().iterrows():
        if r['family'] == 'classical' and r['searched']:
            params.setdefault((r['split'], r['train_cond'], r['model']), r['params'])
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', choices=['in-domain', 'cross-source'], default='in-domain')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--models', nargs='+', default=list(classical.MODELS))
    ap.add_argument('--train-conds', nargs='+', default=['original', 'stripped'])
    args = ap.parse_args()

    df, stats = load_corpus()
    print('Corpus:', stats, flush=True)
    views = {c: text_view(df, c, 'classical') for c in ['original', 'stripped']}
    df = df[['row_id', 'label', 'source']].copy()  # free the raw text; only the cleaned views are needed
    gc.collect()
    y, row_ids = df['label'].to_numpy(), df['row_id'].to_numpy()
    reuse = tuned_params()

    for split_name, seed, split in iter_splits(df, args.setting, args.seeds):
        for train_cond in args.train_conds:
            todo = [m for m in args.models
                    if not all(results.done(split_name, train_cond, tc, m, seed) for tc in TEST_CONDS[train_cond])]
            if not todo:
                continue

            vec = classical.make_vectorizer()
            X_tr = vec.fit_transform([views[train_cond][i] for i in split['train']])
            X_val = vec.transform([views[train_cond][i] for i in split['val']])
            X_te = {tc: vec.transform([views[tc][i] for i in split['test']]) for tc in TEST_CONDS[train_cond]}
            y_tr, y_val, y_te = y[split['train']], y[split['val']], y[split['test']]

            for name in todo:
                key = (split_name, train_cond, name)
                fixed = reuse.get(key) if name in TUNE_ONCE else None
                model, params, val_f1, seconds = classical.fit_tuned(
                    name, X_tr, y_tr, X_val, y_val, seed,
                    params=classical.strip_fitted_only(fixed) if fixed else None)
                if fixed is None:
                    reuse[key] = params

                for tc, X in X_te.items():
                    pred, score = model.predict(X), classical.scores(model, X)
                    results.save_predictions(split_name, train_cond, tc, name, seed,
                                             row_ids[split['test']], y_te, pred, score)
                    m = metrics(y_te, pred, score)
                    results.log_run(family='classical', split=split_name, seed=seed, model=name,
                                    train_cond=train_cond, test_cond=tc, params=params,
                                    searched=fixed is None, val_f1=val_f1, train_seconds=seconds, **m)
                    print(f'{split_name} seed={seed} {train_cond}->{tc} {name:20s} '
                          f"F1={m['f1']:.4f} acc={m['accuracy']:.4f} {params} ({seconds:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
